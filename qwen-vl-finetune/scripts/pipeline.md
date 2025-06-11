# Usage Examples for the Extract Script

# 1. Process specific JSONL files
python extract_batch_inputs.py \
  --input_files mimic-cxr-dataset.jsonl another-dataset.jsonl \
  --output_file batch_inputs.json \
  --samples_per_file 100 \
  --shuffle_output

# 2. Process all JSONL files in a directory
python extract_batch_inputs.py \
  --input_dir ./instruct_datasets \
  --output_file batch_inputs.json \
  --samples_per_file 100 \
  --seed 42

# 3. Process your specific data (based on the sample you provided)
python extract_batch_inputs.py \
  --input_files /path/to/your/instruct_data.jsonl \
  --output_file mimic_cxr_batch_inputs.json \
  --samples_per_file 100

# 4. Then run batch inference with the extracted data
python batch_infer_qwen2_5_vl.py \
  --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
  --processor_dir ./qwen-vl-finetune/output \
  --input_json mimic_cxr_batch_inputs.json \
  --batch_size 8 \
  --max_new_tokens 256 \
  --output_file inference_results.json

# Example of what the extracted JSON will look like:
# {
#   "image_path": "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k/60afa9a2-f372f167-978ebe47-f6417723-bc1f0e08.jpg",
#   "prompt": "I'd like to see a reconstructed scan.",
#   "is_video": false,
#   "source_dataset": "mimic-cxr-dataset.jsonl",
#   "image_id": "60afa9a2-f372f167-978ebe47-f6417723-bc1f0e08"
# }

# Complete workflow:
# Step 1: Extract batch inputs from your instruct datasets
python extract_batch_inputs.py \
  --input_dir ./your_instruct_datasets \
  --output_file batch_inputs.json \
  --samples_per_file 100

# Step 2: Run batch inference
python batch_infer_qwen2_5_vl.py \
  --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
  --processor_dir ./qwen-vl-finetune/output \
  --input_json batch_inputs.json \
  --batch_size 6 \
  --output_file final_results.json

# Step 3: Analyze results
python -c "
import json
with open('final_results.json', 'r') as f:
    results = json.load(f)
success_count = sum(1 for r in results if r['status'] == 'success')
print(f'Successfully processed: {success_count}/{len(results)} samples')
"