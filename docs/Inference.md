## Inference

### Environment Setup

```bash
cd RSCC # path of project root
conda env create -f environment.yaml # genai: env for most baseline models
conda env create -f environment_teochat.yaml # teohat: env for TEOChat
conda env create -f environment_ccexpert.yaml # CCExpert: env for CCExpert
```

### Prepare Pre-trainined Models and Dataset


> [!NOTE]
> As transformers.model_utils `from_pretrained` function would automatically download pre-trained models from huggingface.co, there is the case that you do not have internet connection and would like to use local pre-trained model folder.

We use the same style as [huggingface.co](https://huggingface.co) as `repo_id/model_id`. The model folder should be structured as below:

<details>
<summary>Show Structure</summary>

```text
/path/to/model/folder/ 
├── moonshotai/
│   └── Kimi-VL-A3B-Instruct/
├── Qwen/
│   └── Qwen2-VL-7B-Instruct/
├── Salesforce/
│   └── xgen-mm-phi3-mini-instruct-interleave-r-v1.5/
├── microsoft/
│   └── Phi-4-multimodal-instruct/
├── OpenGVLab/
│   └── InternVL3-8B/
├── llava-hf/
│   ├── llava-interleave-qwen-7b-hf/
│   └── llava-onevision-qwen2-7b-ov-hf/
├── mistralai/
│   └── Pixtral-12B-2409/
├── Meize0729/
│   └── CCExpert_7b/
└── jirvin16/
    └── TEOChat/
```

> [!NOTE]
> When inferencing with BLIP-3 (xgen-mm-phi3-mini-instruct-interleave-r-v1.5) and CCExpert, you may need to pre-download `google/siglip-so400m-patch14-384` under the model folder.
>
> When inference with TEOChat, you may need to pre-download:
> - `LanguageBind/LanguageBind_Image`
> - (Optionally) `LanguageBind/LanguageBind_Video_merge`
>
> Then set in TEOChat's `configs.json`:
> ```json
> {
>   "mm_image_tower": "/path/to/model/folder/LanguageBind/LanguageBind_Image",
>   "mm_video_tower": "/path/to/model/folder/LanguageBind/LanguageBind_Video_merge"
> }
> ```

</details>

Download RSCC dataset and place them under your dataset folder:

```text
/path/to/dataset/folder
├── EBD/
│   └── {events}/
├── xbd/
│   └── images-w512-h512/
│       └── {events}/
└── xbdsubset/
    └── {events}/
```

Set global variable for `PATH_TO_MODEL_FOLDER` and `PATH_TO_DATASET_FOLDER`.

```python
# `RSCC/utils/constants.py`
PATH_TO_MODEL_FOLDER = /path/to/model/folder/ #  "/home/models"
PATH_TO_DATASET_FOLDER = /path/to/dataset/folder # "/home/datasets"
```

### Inference


<details open>
<summary>0. Inference with QvQ-Max</summary>

- Set api configs under `RSCC/.env`.

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

- Run the script.

```python
conda activate genai
python ./inference/xbd_subset_qvq.py
```

</details>

<details>
<summary>1. Inference with baseline models</summary>

> [!WARNING]  
> We support multi-GPUs inference while the Pixtral model and CCExpert model should only be runned on cuda:0.

```python
# inference/xbd_subset_baseline.py
...existing codes...
INFERENCE_MODEL_LIST = [
"moonshotai/Kimi-VL-A3B-Instruct", 
"Qwen/Qwen2-VL-7B-Instruct",
"Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
"microsoft/Phi-4-multimodal-instruct",
"OpenGVLab/InternVL3-8B",
"llava-hf/llava-interleave-qwen-7b-hf",
"llava-hf/llava-onevision-qwen2-7b-ov-hf",
"mistralai/Pixtral-12B-2409",
# "Meize0729/CCExpert_7b", # omit
# "jirvin16/TEOChat", # omit
]
```

```python
conda activate genai
python ./inference/xbd_subset_baseline.py
# or you can speficy the output file path, log file path and device
python ./inference/xbd_subset_baseline.py --output_file "./output/xbd_subset_baseline.jsonl" --log_file "./logs/xbd_subset_baseline.log" --device "cuda:0"
```

</details>

<details>
<summary>2. Inference with TEOChat</summary>

> [!NOTE]  
> The baseline models and specialized model (i.e. TEOChat, CCExpert) use different env. You should use the correspond env along with model_list

```python
# inference/xbd_subset_baseline.py
...existing codes...
INFERENCE_MODEL_LIST = [ "jirvin16/TEOChat"]
```

```bash
conda activate teochat
python ./inference/xbd_subset_baseline.py
# or you can speficy the output file path, log file path and device
```
</details>

<details>
<summary>3. Inference with CCExpert</summary>

> [!NOTE]  
> The baseline models and specialized model (i.e. TEOChat, CCExpert) use different env. You should use the correspond env along with model_list

```python
# inference/xbd_subset_baseline.py
...existing codes...
INFERENCE_MODEL_LIST = [ "Meize0729/CCExpert_7b"]
```

```bash
conda activate CCExpert
python ./inference/xbd_subset_baseline.py
```

</details>

