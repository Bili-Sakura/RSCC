# Additional Inference Models

This guide shows how to run inference with **Skywork-R1V**, **EarthDial** and **NVILA** models using the existing scripts.

## Skywork-R1V

Ensure the checkpoint `Skywork/Skywork-R1V-38B` is placed under `PATH_TO_MODEL_FOLDER`.

```bash
python ./inference/xbd_subset_baseline.py --model_id "Skywork/Skywork-R1V-38B"
```

## EarthDial

Download the weights from Hugging Face (`akshaydudhane/EarthDial_4B_RGB`) and place them under the model folder. Inference can be launched as follows:

```bash
python ./inference/xbd_subset_baseline.py --model_id "akshaydudhane/EarthDial_4B_RGB"
```

## NVILA

NVILA models are loaded via the `llava` backend. Specify the model id `Efficient-Large-Model/NVILA-8B`:

```bash
python ./inference/xbd_subset_baseline.py --model_id "Efficient-Large-Model/NVILA-8B"
```

Alternatively, NVILA's official repository provides a CLI:

```bash
python -m llava.cli.infer --model-path /path/to/NVILA-8B --text "Describe the change" --media pre.png post.png
```
