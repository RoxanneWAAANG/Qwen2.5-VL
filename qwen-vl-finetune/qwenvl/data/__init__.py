import re

# Define placeholders for dataset paths
HEALTHGPT_RECONSTRUCTION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/healthgpt_reconstruct_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k",
}

HEALTHGPT_SUPERRES = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/healthgpt_superres_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k",
}

INTERNET_SEGMENTATION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/internet_seg_dataset.jsonl",
    "data_path": "/home/jack/.cache/kagglehub/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset/versions/10/full-fundus/full-fundus",
}

LLAVA_RAD_REPORT_GENERATION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/llava_rad_rg_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k",
}

LLAVA_SUMMARIZATION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/llava_sum_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/dummy_images",
}

PMC_LLAMA_QA = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/pmc_llama_medqa_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/dummy_images",
}

RATE_NER = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/rate_ner_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/dummy_images",
}

SVLMS_REPORT_GENERATION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/svlms_fundus_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/deepeyenet/deepeyenet/eyenet0420/train_set",
}

ULTRASAM_SEGMENTATION = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/ultrasam_seg_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/UltraSam/dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/images",
}

UNIGRADICON_REG = {
    "annotation_path": "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/tool_instruct/unigradicon_reg_dataset.jsonl",
    "data_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k",
}


data_dict = {
    "healthgpt_reconstruction": HEALTHGPT_RECONSTRUCTION,
    "healthgpt_superres": HEALTHGPT_SUPERRES,
    "internet_segmentation": INTERNET_SEGMENTATION,
    "llava_rad_report_generation": LLAVA_RAD_REPORT_GENERATION,
    "llava_summarization": LLAVA_SUMMARIZATION,
    "pmc_llama_qa": PMC_LLAMA_QA,
    "rate_ner": RATE_NER,
    "svlms_report_generation": SVLMS_REPORT_GENERATION,
    "ultrasam_segmentation": ULTRASAM_SEGMENTATION,
    "unigradicon_reg": UNIGRADICON_REG,
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
