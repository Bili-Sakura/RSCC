# RSCC

**RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events**

Zhenyuan Chen, Chenxi Wang, Ningyu Zhang, Feng Zhang

Zhejiang University

<a href='https://bili-sakura.github.io/RSCC/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://huggingface.co/datasets/BiliSakura/RSCC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://huggingface.co/BiliSakura/RSCCM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>


## Overview

We introduce the Remote Sensing Change Caption (RSCC) dataset, a new benchmark designed to advance the development of large vision-language models for remote sensing. Existing image-text datasets typically rely on single-snapshot imagery and lack the temporal detail crucial for Earth observation tasks. By providing 62,315 pairs of pre-event and post-event images accompanied by detailed change captions, RSCC bridges this gap and enables robust disaster-awareness bi-temporal understanding. We demonstrate its utility through comprehensive experiments using interleaved multimodal large language models. Our results highlight RSCC’s ability to facilitate detailed disaster-related analysis, paving the way for more accurate, interpretable, and scalable vision-language applications in remote sensing. 

<div>
<img src="./assets/rscc_overview2.png" width="1200">
</div>

## 📢News/TODO

**[TODO]** <span style="color:red">Add the latest temporal MLLMs</span>

- [ ] [Skywork-R1V](https://huggingface.co/Skywork)
- [ ] [NVILA](https://huggingface.co/collections/Efficient-Large-Model/nvila-674f8163543890b35a91b428)
- [ ] [EarthDial](https://huggingface.co/akshaydudhane/EarthDial_4B_RGB)

**[IN PROGRESS]** <span style="color:blue">Release RSCC dataset</span>

- [x] <span style="color:gray">2025/05/01</span> All pre-event & post-event images of RSCC (total: 62,315 pairs) are released.  
- [x] <span style="color:gray">2025/05/01</span> The change captions of RSCC-Subset (988 pairs) are released, including 10 baseline model results and QvQ-Max results (ground truth).
- [x] <span style="color:gray">2025/05/01</span> The change captions based on Qwen2.5-VL-72B-Instruct of RSCC (total: 62,315 pairs) are released.
- [ ] Release RSCC change captions based on strong models (e.g., QvQ-Max, o3).

**[COMPLETED]** <span style="color:green">Release code for inference</span>

- [x] <span style="color:gray">2025/05/01</span> Naive inference with baseline models.
- [x] <span style="color:gray">2025/05/15</span> Training-free method augmentation (e.g., VCD, DoLa, DeCo).

**[COMPLETED]** <span style="color:green"> <span style="color:blue">Release RSCCM training scripts</span>

**[COMPLETED]** <span style="color:green">Release code for evaluation</span>

- [x] <span style="color:gray">2025/05/01</span> Metrics for N-Gram (e.g. BLEU, METEOR, ROUGE).
- [x] <span style="color:gray">2025/05/01</span> Metrics for contextual similarity (e.g. Sentence-T5 Similarity, BERTScore).
- [x] <span style="color:gray">2025/05/01</span> Auto comparison of change captions using QvQ-Max (visual reasoning VLM) as a judge.

## Dataset

The dataset can be downloaded from [Huggingface](https://huggingface.co/datasets/BiliSakura/RSCC).  
<div style="display: flex; gap: 20px;">
  <img src="./assets/word_length_distribution.png" alt="Dataset Info" width="500"/>
  <img src="./assets/word_cloud.png" alt="Dataset Info" width="500"/>
</div>



## Benchmark Results

| Model | N-Gram | N-Gram | Contextual Similarity | Contextual Similarity | Avg_L |
|-------|--------|----|----------------------|----|-------|
| (#Activate Params)      | ROUGE(%)↑ | METEOR(%)↑ | BERT(%)↑ | ST5-SCS(%)↑ |  (#Words)   |
| BLIP-3 (3B) | 4.53 | 10.85 | 98.83 | 44.05 | <span style="color:red;">*456</span> |
| &nbsp;&nbsp;+ Textual Prompt | 10.07 (<span style="color:green;">+5.54↑</span>) | 20.69 (<span style="color:green;">+9.84↑</span>) | 98.95 (<span style="color:green;">+0.12↑</span>) | 63.67 (<span style="color:green;">+19.62↑</span>) | <span style="color:red;">*302</span> |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 8.45 (<span style="color:red;">-1.62↓</span>) | 19.18 (<span style="color:red;">-1.51↓</span>) | 99.01 (<span style="color:green;">+0.06↑</span>) | 68.34 (<span style="color:green;">+4.67↑</span>) | <span style="color:red;">*354</span> |
| Kimi-VL (3B) | 12.47 | 16.95 | 98.83 | 51.35 | 87 |
| &nbsp;&nbsp;+ Textual Prompt | 16.83 (<span style="color:green;">+4.36↑</span>) | 25.47 (<span style="color:green;">+8.52↑</span>) | 99.22 (<span style="color:green;">+0.39↑</span>) | 70.75 (<span style="color:green;">+19.40↑</span>) | 108 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 16.83 (+0.00) | 25.39 (<span style="color:red;">-0.08↓</span>) | 99.30 (<span style="color:green;">+0.08↑</span>) | 69.97 (<span style="color:red;">-0.78↓</span>) | 109 |
| Phi-4-Multimodal (4B) | 4.09 | 1.45 | 98.60 | 34.55 | 7 |
| &nbsp;&nbsp;+ Textual Prompt | 17.08 (<span style="color:green;">+13.00↑</span>) | 19.70 (<span style="color:green;">+18.25↑</span>) | 98.93 (<span style="color:green;">+0.33↑</span>) | 67.62 (<span style="color:green;">+33.07↑</span>) | 75 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 17.05 (<span style="color:red;">-0.03↓</span>) | 19.09 (<span style="color:red;">-0.61↓</span>) | 98.90 (<span style="color:red;">-0.03↓</span>) | 66.69 (<span style="color:red;">-0.93↓</span>) | 70 |
| Qwen2-VL (7B) | 11.02 | 9.95 | 99.11 | 45.55 | 42 |
| &nbsp;&nbsp;+ Textual Prompt | 19.04 (<span style="color:green;">+8.02↑</span>) | 25.20 (<span style="color:green;">+15.25↑</span>) | 99.01 (<span style="color:red;">-0.10↓</span>) | 72.65 (<span style="color:green;">+27.10↑</span>) | 84 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 18.43 (<span style="color:red;">-0.61↓</span>) | 25.03 (<span style="color:red;">-0.17↓</span>) | 99.03 (<span style="color:green;">+0.02↑</span>) | 72.89 (<span style="color:green;">+0.24↑</span>) | 88 |
| LLaVA-NeXT-Interleave (8B) | 12.51 | 13.29 | 99.11 | 46.99 | 57 |
| &nbsp;&nbsp;+ Textual Prompt | 16.09 (<span style="color:green;">+3.58↑</span>) | 20.73 (<span style="color:green;">+7.44↑</span>) | 99.22 (<span style="color:green;">+0.11↑</span>) | 62.60 (<span style="color:green;">+15.61↑</span>) | 75 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 15.76 (<span style="color:red;">-0.33↓</span>) | 21.17 (<span style="color:green;">+0.44↑</span>) | 99.24 (<span style="color:green;">+0.02↑</span>) | 65.75 (<span style="color:green;">+3.15↑</span>) | 88 |
| LLaVA-OneVision (8B) | 8.40 | 10.97 | 98.64 | 46.15 | <span style="color:red;">*221</span> |
| &nbsp;&nbsp;+ Textual Prompt | 11.15 (<span style="color:green;">+2.75↑</span>) | 19.09 (<span style="color:green;">+8.12↑</span>) | 98.85 (<span style="color:green;">+0.21↑</span>) | 70.08 (<span style="color:green;">+23.93↑</span>) | <span style="color:red;">*285</span> |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 10.68 (<span style="color:red;">-0.47↓</span>) | 18.27 (<span style="color:red;">-0.82↓</span>) | 98.79 (<span style="color:red;">-0.06↓</span>) | 69.34 (<span style="color:red;">-0.74↓</span>) | <span style="color:red;">*290</span> |
| InternVL 3 (8B) | 12.76 | 15.77 | 99.31 | 51.84 | 64 |
| &nbsp;&nbsp;+ Textual Prompt | _19.81_ (<span style="color:green;">+7.05↑</span>) | _28.51_ (<span style="color:green;">+12.74↑</span>) | **99.55** (<span style="color:green;">+0.24↑</span>) | 78.57 (<span style="color:green;">+26.73↑</span>) | 81 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 19.70 (<span style="color:red;">-0.11↓</span>) | 28.46 (<span style="color:red;">-0.05↓</span>) | 99.51 (<span style="color:red;">-0.04↓</span>) | **79.18** (<span style="color:green;">+0.61↑</span>) | 84 |
| Pixtral (12B) | 12.34 | 15.94 | 99.34 | 49.36 | 70 |
| &nbsp;&nbsp;+ Textual Prompt | **19.87** (<span style="color:green;">+7.53↑</span>) | **29.01** (<span style="color:green;">+13.07↑</span>) | 99.51 (<span style="color:green;">+0.17↑</span>) | _79.07_ (<span style="color:green;">+29.71↑</span>) | 97 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 19.03 (<span style="color:red;">-0.84↓</span>) | 28.44 (<span style="color:red;">-0.57↓</span>) | _99.52_ (<span style="color:green;">+0.01↑</span>) | 78.71 (<span style="color:red;">-0.36↓</span>) | 102 |
| CCExpert (7B) | 7.61 | 4.32 | 99.17 | 40.81 | 12 |
| &nbsp;&nbsp;+ Textual Prompt | 8.71 (<span style="color:green;">+1.10↑</span>) | 5.35 (<span style="color:green;">+1.03↑</span>) | 99.23 (<span style="color:green;">+0.06↑</span>) | 47.13 (<span style="color:green;">+6.32↑</span>) | 14 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 8.84 (<span style="color:green;">+0.13↑</span>) | 5.41 (<span style="color:green;">+0.06↑</span>) | 99.23 (+0.00) | 46.58 (<span style="color:red;">-0.55↓</span>) | 14 |
| TEOChat (7B) | 7.86 | 5.77 | 98.99 | 52.64 | 15 |
| &nbsp;&nbsp;+ Textual Prompt | 11.81 (<span style="color:green;">+3.95↑</span>) | 10.24 (<span style="color:green;">+4.47↑</span>) | 99.12 (<span style="color:green;">+0.13↑</span>) | 61.73 (<span style="color:green;">+9.09↑</span>) | 22 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Visual Prompt | 11.55 (<span style="color:red;">-0.26↓</span>) | 10.04 (<span style="color:red;">-0.20↓</span>) | 99.09 (<span style="color:red;">-0.03↓</span>) | 62.53 (<span style="color:green;">+0.80↑</span>) | 22 |
