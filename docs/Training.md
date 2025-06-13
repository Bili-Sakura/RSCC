
### Fine-tuning RSCCM

```bash
cd RSCC
conda env create -f environment_qwenvl_ft.yaml
conda activate qwenvl_ft
bash train/qwen-vl-finetune/scripts/sft_for_rscc_model.sh
```

### Auto Comparison with MLLMs (e.g. Qwen QvQ-Max)

We provide scripts that employ the latest visual reasoning proprietary model (QvQ-Max) to choose the best change caption from a series of candidates.

<details>
<summary>Show Steps</summary>

1. Set api configs under `RSCC/.env`.

```env
# API key for DashScope (keep this secret!)
DASHSCOPE_API_KEY="sk-xxxxxxxxxx"  

# Model ID should match the official code
QVQ_MODEL_NAME="qvq-max-2025-03-25"  

# API base URL
API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  

# Maximum concurrent workers
MAX_WORKERS=30  

# Token threshold warning level
TOKEN_THRESHOLD=10000  
```

2. Run the script.

```bash
conda activate genai
python ./evaluation/autoeval.py
```

The token usage is auto logged and you can also check `RSCC/data/token_usage.json` to keep update with remaining token number.

</details>

## Licensing Information

The dataset is released under the [CC-BY-4.0]([https://creativecommons.org/licenses/by-nc/4.0/deed.en](https://creativecommons.org/licenses/by/4.0/deed.en)), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.



## 🙏 Acknowledgement

Our RSCC dataset is built based on [xBD](https://www.xview2.org/) and [EBD](https://figshare.com/articles/figure/An_Extended_Building_Damage_EBD_dataset_constructed_from_disaster-related_bi-temporal_remote_sensing_images_/25285009) datasets.

We are thankful to [Kimi-VL](https://hf-mirror.com/moonshotai/Kimi-VL-A3B-Instruct), [BLIP-3](https://hf-mirror.com/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5), [Phi-4-Multimodal](https://hf-mirror.com/microsoft/Phi-4-multimodal-instruct), [Qwen2-VL](https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct), [Qwen2.5-VL](https://hf-mirror.com/Qwen/Qwen2.5-VL-72B-Instruct), [LLaVA-NeXT-Interleave](https://hf-mirror.com/llava-hf/llava-interleave-qwen-7b-hf),[LLaVA-OneVision](https://hf-mirror.com/llava-hf/llava-onevision-qwen2-7b-ov-hf), [InternVL 3](https://hf-mirror.com/OpenGVLab/InternVL3-8B), [Pixtral](https://hf-mirror.com/mistralai/Pixtral-12B-2409), [TEOChat](https://github.com/ermongroup/TEOChat) and [CCExpert](https://github.com/Meize0729/CCExpert) for releasing their models and code as open-source contributions.

The metrics implements are derived from [huggingface/evaluate](https://github.com/huggingface/evaluate).

The training implements are derived from [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).

## 📜 Citation

```bibtex
@article{rscc_chen_2025,
  title={RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events},
  author={Zhenyuan Chen, Chenxi Wang, Ningyu Zhang, Feng Zhang},
  year={2025},
  howpublished={\url{https://github.com/Bili-Sakura/RSCC}}
}
