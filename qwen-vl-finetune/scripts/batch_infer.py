#!/usr/bin/env python3
"""
Batch inference script for Qwen2.5-VL with separate checkpoint and processor directories.

python3 ./scripts/batch_infer.py \
  --ckpt_dir /home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/output_7b/checkpoint-160 \
  --processor_dir /home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/output_7b \
  --input_json ./scripts/batch_inputs.json \
  --batch_size 4 \
  --output_file ./scripts/final_results.json
"""

# export PYTHONPATH=$PYTHONPATH:/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-utils/src

import argparse
import json
import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Base model identifier for chat_template fallback
BASE_MODEL_NAME = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen2.5-VL-7B-Instruct"

# Supported image/video extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}


def load_model_and_processor(ckpt_dir: str, processor_dir: str):
    """
    Load model weights from ckpt_dir and processor configuration from processor_dir.
    If the loaded processor lacks a chat_template, fetch it from the base model.
    """
    print("Loading model and processor...")
    # Load fine-tuned model weights
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load (possibly incomplete) processor configs
    processor = AutoProcessor.from_pretrained(processor_dir)

    # Set padding side to left for decoder-only models
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'
        print("Set tokenizer padding_side to 'left' for decoder-only model")

    # Ensure chat_template is set (needed for apply_chat_template)
    if not hasattr(processor, 'chat_template') or processor.chat_template is None:
        print("Loading chat_template from base model...")
        base_proc = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
        processor.chat_template = base_proc.chat_template

    print(f"Model loaded on device: {model.device}")
    return model, processor


def build_messages(image_path: str, prompt: str, is_video: bool = False):
    """
    Build the chat-style messages payload from image/video and prompt.
    """
    content_item = {"type": "video", "video": image_path} if is_video else {"type": "image", "image": image_path}
    return [
        {"role": "user", "content": [content_item, {"type": "text", "text": prompt}]}  
    ]


def run_batch_inference(model, processor, batch_data: List[Dict], max_new_tokens: int = 128):
    """
    Run batch inference on multiple inputs.
    
    Args:
        model: The loaded Qwen2.5-VL model
        processor: The processor
        batch_data: List of dicts with keys: 'image_path', 'prompt', 'is_video'
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of generated responses
    """
    if not batch_data:
        return []
    
    # Prepare all messages
    all_messages = []
    for item in batch_data:
        messages = build_messages(item['image_path'], item['prompt'], item.get('is_video', False))
        all_messages.append(messages)
    
    # Prepare chat prompts
    texts = []
    all_image_inputs = []
    all_video_inputs = []
    
    for messages in all_messages:
        # Get chat template text
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
        
        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)
        all_image_inputs.extend(image_inputs or [])
        all_video_inputs.extend(video_inputs or [])
    
    # Tokenize and batch process
    try:
        inputs = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate with the model
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Remove prompt tokens from output
        response_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode responses
        outputs = processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return outputs
        
    except Exception as e:
        print(f"Error during batch inference: {e}")
        # Fallback to individual processing
        print("Falling back to individual processing...")
        outputs = []
        for item in batch_data:
            try:
                result = run_single_inference(model, processor, item, max_new_tokens)
                outputs.append(result)
            except Exception as single_e:
                print(f"Error processing {item['image_path']}: {single_e}")
                outputs.append(f"Error: {str(single_e)}")
        return outputs


def run_single_inference(model, processor, item_data: Dict, max_new_tokens: int = 128):
    """
    Run inference on a single item (fallback for batch failures).
    """
    messages = build_messages(item_data['image_path'], item_data['prompt'], item_data.get('is_video', False))
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    response_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output[0]


def load_inputs_from_csv(csv_path: str) -> List[Dict]:
    """Load inputs from CSV file."""
    print(f"Loading inputs from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    required_columns = ['image_path', 'prompt']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    inputs = []
    for _, row in df.iterrows():
        item = {
            'image_path': row['image_path'],
            'prompt': row['prompt'],
            'is_video': bool(row.get('is_video', False))
        }
        inputs.append(item)
    
    print(f"Loaded {len(inputs)} inputs from CSV")
    return inputs


def load_inputs_from_json(json_path: str) -> List[Dict]:
    """Load inputs from JSON file."""
    print(f"Loading inputs from JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects")
    
    inputs = []
    for item in data:
        if 'image_path' not in item or 'prompt' not in item:
            raise ValueError("Each JSON object must have 'image_path' and 'prompt' fields")
        inputs.append({
            'image_path': item['image_path'],
            'prompt': item['prompt'],
            'is_video': item.get('is_video', False)
        })
    
    print(f"Loaded {len(inputs)} inputs from JSON")
    return inputs


def load_inputs_from_directory(dir_path: str, prompt: str, include_videos: bool = True) -> List[Dict]:
    """Load all images/videos from directory with the same prompt."""
    print(f"Loading inputs from directory: {dir_path}")
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    inputs = []
    extensions = IMAGE_EXTENSIONS | (VIDEO_EXTENSIONS if include_videos else set())
    
    for file_path in dir_path.rglob('*'):
        if file_path.suffix.lower() in extensions:
            is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS
            inputs.append({
                'image_path': str(file_path),
                'prompt': prompt,
                'is_video': is_video
            })
    
    inputs.sort(key=lambda x: x['image_path'])  # Sort for consistent ordering
    print(f"Found {len(inputs)} files in directory")
    return inputs


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def process_in_batches(model, processor, all_inputs: List[Dict], batch_size: int, max_new_tokens: int) -> List[Dict]:
    """Process inputs in batches with progress tracking."""
    results = []
    
    print(f"Processing {len(all_inputs)} inputs in batches of {batch_size}")
    
    for i in tqdm(range(0, len(all_inputs), batch_size), desc="Processing batches"):
        batch = all_inputs[i:i + batch_size]
        
        try:
            batch_outputs = run_batch_inference(model, processor, batch, max_new_tokens)
            
            # Combine inputs with outputs
            for inp, output in zip(batch, batch_outputs):
                result = {
                    'image_path': inp['image_path'],
                    'prompt': inp['prompt'],
                    'is_video': inp.get('is_video', False),
                    'response': output,
                    'status': 'success'
                }
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add error results for this batch
            for inp in batch:
                result = {
                    'image_path': inp['image_path'],
                    'prompt': inp['prompt'],
                    'is_video': inp.get('is_video', False),
                    'response': f"Error: {str(e)}",
                    'status': 'error'
                }
                results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Qwen2.5-VL inference script")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to the checkpoint directory containing model weights")
    parser.add_argument("--processor_dir", type=str, required=True,
                        help="Path to the processor config directory")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_csv", type=str,
                           help="Path to CSV file with columns: image_path, prompt, is_video")
    input_group.add_argument("--input_json", type=str,
                           help="Path to JSON file with list of input objects")
    input_group.add_argument("--input_dir", type=str,
                           help="Path to directory containing images/videos")
    
    # Additional arguments
    parser.add_argument("--prompt", type=str,
                        help="Prompt to use for all files (required when using --input_dir)")
    parser.add_argument("--output_file", type=str, default="batch_results.json",
                        help="Path to save results JSON file")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing (adjust based on GPU memory)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--include_videos", action="store_true", default=True,
                        help="Include videos when processing directory (default: True)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_dir and not args.prompt:
        parser.error("--prompt is required when using --input_dir")
    
    # Load model and processor
    model, processor = load_model_and_processor(args.ckpt_dir, args.processor_dir)
    
    # Load inputs based on chosen method
    if args.input_csv:
        inputs = load_inputs_from_csv(args.input_csv)
    elif args.input_json:
        inputs = load_inputs_from_json(args.input_json)
    elif args.input_dir:
        inputs = load_inputs_from_directory(args.input_dir, args.prompt, args.include_videos)
    
    if not inputs:
        print("No inputs found!")
        return
    
    # Process inputs in batches
    results = process_in_batches(model, processor, inputs, args.batch_size, args.max_new_tokens)
    
    # Save results
    save_results(results, args.output_file)
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"\nBatch processing completed!")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
