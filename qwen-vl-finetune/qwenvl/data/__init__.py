import re

# Define placeholders for dataset paths
HEALTHGPT_RECONSTRUCTION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/healthgpt_reconstruct_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k",
}

data_dict = {
    "healthgpt_reconstruction": HEALTHGPT_RECONSTRUCTION,
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


if __name__ == "__main__":
    dataset_names = ["healthgpt_reconstruction"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
