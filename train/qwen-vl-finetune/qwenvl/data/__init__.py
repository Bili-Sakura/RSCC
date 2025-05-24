import re

# Define placeholders for dataset paths

RSCC_DATASET = {
    "annotation_path": "/home/sakura/projects/RSCC/train/qwen-vl-finetune/mydataset/rscc_qwenvl_format.jsonl",
    "data_path": "",  # Absolute paths already in JSONL
}

RSCC_SUBSET_DATASET = {
    "annotation_path": "/home/sakura/projects/RSCC/train/qwen-vl-finetune/mydataset/rscc_subset_qwenvl_format.jsonl",
    "data_path": "",  # Absolute paths already in JSONL
}

CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    # "cambrian_737k": CAMBRIAN_737K,
    # "mp_doc": MP_DOC,
    # "clevr_mc": CLEVR_MC,
    # "videochatgpt": VIDEOCHATGPT,
    "rscc_subset": RSCC_SUBSET_DATASET,  # Add this line
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


# Add path validation at the bottom
if __name__ == "__main__":
    # Test RSCC dataset config
    dataset_names = ["rscc%100"]  # Changed from cambrian_737k to test your dataset
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
        # Verify the annotation file exists
        import os

        if not os.path.exists(config["annotation_path"]):
            raise FileNotFoundError(
                f"Annotation file not found: {config['annotation_path']}"
            )
