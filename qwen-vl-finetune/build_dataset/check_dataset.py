with open("build_dataset/tool_instruct/llava_sum_dataset.jsonl", "r", encoding="utf-8") as f:
    first_line = f.readline().rstrip("\n")
    print(first_line)
