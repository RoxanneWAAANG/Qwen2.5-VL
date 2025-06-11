Batch inference script for Qwen2.5-VL with separate checkpoint and processor directories.
Supports multiple input methods: CSV file, directory processing, or JSON file.

Usage Examples:
    # Process from CSV file
    python batch_infer_qwen2_5_vl.py \
      --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
      --processor_dir ./qwen-vl-finetune/output \
      --input_csv inputs.csv \
      --output_file results.json

    # Process directory of images with same prompt
    python batch_infer_qwen2_5_vl.py \
      --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
      --processor_dir ./qwen-vl-finetune/output \
      --input_dir ./images \
      --prompt "Describe this image." \
      --output_file results.json

    # Process from JSON file
    python batch_infer_qwen2_5_vl.py \
      --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
      --processor_dir ./qwen-vl-finetune/output \
      --input_json inputs.json \
      --output_file results.json

CSV Format:
    image_path,prompt,is_video
    path/to/image1.jpg,"Describe this image",false
    path/to/video1.mp4,"What happens in this video?",true

JSON Format:
    [
        {"image_path": "path/to/image1.jpg", "prompt": "Describe this image", "is_video": false},
        {"image_path": "path/to/video1.mp4", "prompt": "What happens in this video?", "is_video": true}
    ]

Prerequisites:
```python
    pip install transformers==4.51.3 accelerate qwen-vl-utils[decord] pandas tqdm
```

# Example CSV file (inputs.csv)
image_path,prompt,is_video
./images/medical_scan1.jpg,"Describe the medical findings in this image",false
./images/xray_chest.png,"What abnormalities can you see in this chest X-ray?",false
./videos/surgery_clip.mp4,"Summarize the surgical procedure shown",true
./images/pathology_slide.jpg,"Identify the tissue type and any pathological features",false

# Example JSON file (inputs.json)
[
  {
    "image_path": "./images/medical_scan1.jpg",
    "prompt": "Describe the medical findings in this image",
    "is_video": false
  },
  {
    "image_path": "./images/xray_chest.png", 
    "prompt": "What abnormalities can you see in this chest X-ray?",
    "is_video": false
  },
  {
    "image_path": "./videos/surgery_clip.mp4",
    "prompt": "Summarize the surgical procedure shown",
    "is_video": true
  }
]

# Command examples:

# Process from CSV:
python batch_infer_qwen2_5_vl.py \
  --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
  --processor_dir ./qwen-vl-finetune/output \
  --input_csv inputs.csv \
  --batch_size 8 \
  --output_file medical_results.json

# Process directory with same prompt:
python batch_infer_qwen2_5_vl.py \
  --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
  --processor_dir ./qwen-vl-finetune/output \
  --input_dir ./medical_images \
  --prompt "Analyze this medical image and provide a detailed description" \
  --batch_size 4 \
  --output_file batch_medical_analysis.json

# Process from JSON:
python batch_infer_qwen2_5_vl.py \
  --ckpt_dir ./qwen-vl-finetune/output/checkpoint-78 \
  --processor_dir ./qwen-vl-finetune/output \
  --input_json inputs.json \
  --batch_size 6 \
  --max_new_tokens 256 \
  --output_file detailed_results.json