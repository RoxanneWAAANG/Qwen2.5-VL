import json

def jsonl_to_json(input_file: str, output_file: str):
    """
    将JSONL文件转换为JSON数组格式
    """
    data = []
    
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    print(f"Line content: {line[:100]}...")
                    continue
    
    print(f"Loaded {len(data)} records")
    print(f"Writing to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed! {len(data)} records saved to {output_file}")

def json_to_jsonl(input_file: str, output_file: str):
    """
    将JSON数组格式转换为JSONL文件（反向操作）
    """
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print("Error: JSON file should contain an array of objects")
        return
    
    print(f"Writing to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed! {len(data)} records saved to {output_file}")

# 使用示例
if __name__ == "__main__":
    # JSONL -> JSON
    jsonl_to_json(
        input_file="/home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Qwen2.5-VL/qwen-vl-finetune/output_qwen/error_samples.jsonl", 
        output_file="/home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Qwen2.5-VL/qwen-vl-finetune/output_qwen/error_samples.json"
    )
    
    # 如果需要反向转换 JSON -> JSONL
    # json_to_jsonl(
    #     input_file="test.json",
    #     output_file="test_converted.jsonl" 
    # )