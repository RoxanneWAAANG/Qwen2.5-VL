import re

# Define placeholders for dataset paths
SINGLE_TOOL_MULTIROUND = {
    "annotation_path": "build_dataset/multi_round/single_tool_multiround.jsonl",
    "data_path": "",
}

MULTI_TOOL_MULTIROUND = {
    "annotation_path": "build_dataset/multi_round/multi_tool_multiround.jsonl",
    "data_path": "",
}

MULTI_TOOL_SINGLE_ROUND = {
    "annotation_path": "build_dataset/multi_round/multi_tool_single_round.jsonl",
    "data_path": "",
}


data_dict = {
    "single_tool_multi_round": SINGLE_TOOL_MULTIROUND,
    "multi_tool_multi_round": MULTI_TOOL_MULTIROUND,
    "multi_tool_single_round": MULTI_TOOL_SINGLE_ROUND,
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
