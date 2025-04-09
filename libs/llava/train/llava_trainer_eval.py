
import json
import math
from tqdm import tqdm
import string
import subprocess
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    EvalPrediction,
    speed_metrics,
)
from transformers.trainer_utils import (
    has_length
)
from transformers.utils import logging
import mmengine.dist as dist_mm

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

from llava.train.llava_trainer import LLaVATrainer
from llava.utils import rank0_print

logger = logging.get_logger(__name__)

def get_eval_score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),
    ]
    # [['the airport is very large', 'next to the airport is green grass', 'next to the airport is green grass', 'the airport is very large', 'the airport is very large'], ['the airport is very large', 'next to the airport is green grass', 'next to the airport is green grass', 'the airport is very large', 'the airport is very large']]
    # [['3 airplane wings sitting on top of the ground'], ['3 airplane wings sitting on top of the ground']]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    
    score_dict = dict(zip(method, score))
    score_dict['Sm'] = (score_dict['Bleu_4'] + score_dict['CIDEr'] + score_dict['METEOR'] + score_dict['ROUGE_L']) / 4
    return score_dict

class EvalLoopContainer:
    def __init__(self,):
        self.results = []

    def add(self, tensors) -> None:
        self.results.append(tensors)

    def get_results(self):
        return self.results

class EvalLoopOutput(NamedTuple):
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

class LLaVATrainerEvalNew(LLaVATrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics based on model.generate.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                The evaluation dataset.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model that should be ignored.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix.
        """
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        output = self.generate_loop(eval_dataloader, description="Evaluation", metric_key_prefix=metric_key_prefix)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def generate_loop(
        self,
        dataloader: DataLoader,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Generate loop based on model.generate for evaluation.
        """
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        batch_size = self.args.eval_batch_size

        rank0_print(f"\n***** Running {description} *****")
        if has_length(dataloader):
            rank0_print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            rank0_print("  Num examples: Unknown")
        rank0_print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)
        if args.past_index >= 0:
            self._past = None
        
        # =========================== Important Part ===========================
        keep_in_mind_dtype = model.dtype
        model.to(dtype=torch.float32)
        all_preds_gts = EvalLoopContainer()
        
        observed_num_examples = 0

        tmp_results = []
        # import pdb; pdb.set_trace()
        for step, inputs in enumerate(dataloader):
            observed_num_examples += 1
            # Prepare inputs
            input_ids = inputs["input_ids"]
            # get "assistant" token
            assistant_index = (input_ids[0] == 77091).nonzero(as_tuple=True)[0][-1].item()
            input_ids = input_ids[:, :assistant_index+2]

            images = [image_tensor.to(model.dtype) for image_tensor in inputs["images"]]
            image_sizes = inputs["image_sizes"]
            label = inputs["extra_info"][0]["change_caption"]
            uid = inputs["uid"][0]
            with torch.no_grad():
                # Generate predictions
                generated_tokens = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                )
            # generated_tokens = self.gather_function((generated_tokens))
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            tmp_pred_gt = dict(
                pred=[decoded_preds],
                gt=label,
                uid=uid,
            )
            tmp_results.append(tmp_pred_gt)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1  # Default to 1 if not using distributed training


        all_pred_gt = dist_mm.collect_results(tmp_results, world_size*observed_num_examples, 'cpu')
        # all_preds_gts = torch.distributed.all_gather(all_preds_gts)
        if torch.distributed.get_rank() == 0:
            all_preds = []
            all_gts = []
            uid_set = set()
            for pred_gt in all_pred_gt:
                if pred_gt['uid'] not in uid_set:
                    uid_set.add(pred_gt['uid'])
                    all_preds.append(pred_gt['pred'])
                    all_gts.append(pred_gt['gt'])
            assert len(all_preds) == num_samples
        
            def process_text_list(text_list):
                processed_list = []
                for sublist in text_list:
                    processed_sublist = []
                    for sentence in sublist:
                        # 将字符串转换为小写并去除句号
                        sentence = sentence.lower().replace('.', '')
                        processed_sublist.append(sentence)
                    processed_list.append(processed_sublist)
                return processed_list

            score_metrics = get_eval_score(process_text_list(all_gts), process_text_list(all_preds))
            score_metrics_list_to_broadcast = [{f"{metric_key_prefix}_{key}": value for key, value in score_metrics.items()}]
        else:
            score_metrics_list_to_broadcast = [None]
        dist_mm.broadcast_object_list(score_metrics_list_to_broadcast)
        model.to(dtype=keep_in_mind_dtype)
        return EvalLoopOutput(metrics=score_metrics_list_to_broadcast[0], num_samples=num_samples)

class LLaVAEvalTrainer(LLaVATrainer):
    def evaluate(self, evaluate_args):
        cmd = f"accelerate launch --num_processes {evaluate_args.eval_num_processes} -m lmms_eval \
                --model {evaluate_args.model} \
                --model_args {evaluate_args.model_args} \
                --tasks {evaluate_args.task_names} \
                --batch_size {evaluate_args.batch_size} \
                --log_samples_suffix {evaluate_args.log_samples_suffix} \
                --output_path {evaluate_args.output_path}"
        if evaluate_args.limit:
            cmd += f" --limit {evaluate_args.limit}"
        if evaluate_args.num_fewshot:
            cmd += f" --num_fewshot {evaluate_args.num_fewshot}"
        if evaluate_args.gen_kwargs != "":
            cmd += f" --gen_kwargs {evaluate_args.gen_kwargs}"
        if evaluate_args.log_samples:
            cmd += f" --log_samples"
        else:
            assert False, "Please log samples so that the result can be parsed"
        results = subprocess.run([cmd], shell=True, capture_output=True, text=True)
        try:
            result_file_index_start = results.stdout.index("Saved samples to ")
            result_file_index_end = results.stdout.index(f".json")
            result_file_index_start += len("Saved samples to ")
            file = results.stdout[result_file_index_start:result_file_index_end]
        except:
            result_file_index_start = results.stderr.index("Saved samples to ")
            result_file_index_end = results.stderr.index(f".json")
            result_file_index_start += len("Saved samples to ")
            file = results.stderr[result_file_index_start:result_file_index_end]
        file = file.split("/")[:-1]
        file = "/".join(file) + "/results.json"
        with open(file, "r") as f:
            lmms_eval_results = json.load(f)
        result_dict = {}
        tasks_list = evaluate_args.task_names.split(",")
        for task in tasks_list:
            task_results = lmms_eval_results["results"][task]
            for k, v in task_results.items():
                if k != "alias" and "stderr" not in k:
                    metric = k.split(",")[0]
                    result_dict[f"{task}_{metric}"] = v
        return result_dict

    """def evaluate(self, evaluate_args):
        initialize_tasks()
        tasks_list = evaluate_args.task_names.split(",")
        result_dict = {}
        results = evaluator.simple_evaluate(
            model=evaluate_args.model,
            model_args=evaluate_args.model_args,
            tasks=tasks_list,
            num_fewshot=evaluate_args.num_fewshot,
            batch_size=evaluate_args.batch_size,
            device=evaluate_args.device,
            limit=evaluate_args.limit,
            check_integrity=evaluate_args.check_integrity,
            show_task_to_terminal=evaluate_args.show_task_to_terminal,
            log_samples=evaluate_args.log_samples,
            gen_kwargs=evaluate_args.gen_kwargs,
            cli_args=evaluate_args,
        )
        for task in tasks_list:
            task_results = results["results"][task]
            for k,v in task_results.items():
                if k != "alias" and "stderr" not in k:
                    metric = k.split(",")[0]
                    result_dict[f"{task}_{metric}"] = v
            
        return result_dict"""


if __name__ == '__main__':
    pass