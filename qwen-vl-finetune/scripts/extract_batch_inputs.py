#!/usr/bin/env python3
"""
Script to extract batch inference inputs from instruct datasets.
Converts JSONL instruct format to batch inference JSON format.

Usage:
    python3 ./scripts/extract_batch_inputs.py \
      --input_dir ./build_dataset/tool_instruct \
      --output_file batch_inputs.json \
      --samples_per_file 100
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
import os


def clean_prompt(prompt_text: str) -> str:
    """
    Clean the prompt text by removing image tokens and extra whitespace.
    """
    # Remove <image> tokens
    cleaned = re.sub(r'<image>\s*', '', prompt_text)
    
    # Remove extra whitespace and newlines
    cleaned = ' '.join(cleaned.split())
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def extract_human_prompt(conversations: List[Dict]) -> str:
    """
    Extract the human's prompt from the conversation.
    Returns the first human message with <image> token removed.
    """
    for conv in conversations:
        if conv.get('from') == 'human':
            prompt = conv.get('value', '')
            return clean_prompt(prompt)
    
    return ""


def process_jsonl_file(file_path: str, samples_per_file: int, dataset_name: str = None) -> List[Dict]:
    """
    Process a JSONL file and extract samples for batch inference.
    
    Args:
        file_path: Path to the JSONL file
        samples_per_file: Number of samples to extract
        dataset_name: Name to use for source dataset field
        
    Returns:
        List of batch inference input dictionaries
    """
    if dataset_name is None:
        dataset_name = Path(file_path).name
    
    print(f"Processing {file_path}...")
    
    samples = []
    valid_samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract required fields
                    image_path = data.get('file_name', '')
                    image_id = data.get('image_id', '')
                    conversations = data.get('conversations', [])
                    
                    # Extract human prompt
                    prompt = extract_human_prompt(conversations)
                    
                    # Validate the sample
                    if not image_path or not prompt:
                        print(f"  Skipping line {line_num}: missing image_path or prompt")
                        continue
                    
                    # Check if image file exists (optional)
                    if not os.path.exists(image_path):
                        print(f"  Warning: Image file not found: {image_path}")
                        # Continue anyway, user might want to process even if files are missing
                    
                    # Determine if it's a video (based on file extension)
                    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
                    is_video = Path(image_path).suffix.lower() in video_extensions
                    
                    sample = {
                        'image_path': image_path,
                        'prompt': prompt,
                        'is_video': is_video,
                        'source_dataset': dataset_name,
                        'image_id': image_id
                    }
                    
                    valid_samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    print(f"  Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"  Error processing line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    print(f"  Found {len(valid_samples)} valid samples")
    
    # Randomly sample the requested number
    if len(valid_samples) > samples_per_file:
        samples = random.sample(valid_samples, samples_per_file)
        print(f"  Randomly selected {samples_per_file} samples")
    else:
        samples = valid_samples
        print(f"  Using all {len(samples)} samples (less than requested {samples_per_file})")
    
    return samples


def find_jsonl_files(directory: str) -> List[str]:
    """
    Find all JSONL files in a directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    jsonl_files = []
    for pattern in ['*.jsonl', '*.json']:
        jsonl_files.extend(dir_path.glob(pattern))
    
    return [str(f) for f in jsonl_files]


def main():
    parser = argparse.ArgumentParser(description="Extract batch inference inputs from instruct datasets")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_files", nargs='+', type=str,
                           help="List of JSONL files to process")
    input_group.add_argument("--input_dir", type=str,
                           help="Directory containing JSONL files")
    
    parser.add_argument("--output_file", type=str, default="batch_inputs.json",
                        help="Output JSON file for batch inference")
    parser.add_argument("--samples_per_file", type=int, default=100,
                        help="Number of samples to extract per file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--shuffle_output", action="store_true",
                        help="Shuffle the final output samples")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Get list of files to process
    if args.input_files:
        input_files = args.input_files
    else:
        input_files = find_jsonl_files(args.input_dir)
        if not input_files:
            print(f"No JSONL files found in directory: {args.input_dir}")
            return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    all_samples = []
    for file_path in input_files:
        samples = process_jsonl_file(file_path, args.samples_per_file, Path(file_path).name)
        all_samples.extend(samples)
    
    if not all_samples:
        print("No valid samples extracted!")
        return
    
    # Shuffle output if requested
    if args.shuffle_output:
        random.shuffle(all_samples)
        print("Shuffled output samples")
    
    # Save to output file
    print(f"Saving {len(all_samples)} samples to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nExtraction completed!")
    print(f"Total samples: {len(all_samples)}")
    
    # Print source breakdown
    source_counts = {}
    for sample in all_samples:
        source = sample['source_dataset']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nSamples per dataset:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Print some example prompts
    print(f"\nExample prompts:")
    for i, sample in enumerate(all_samples[:3]):
        print(f"  {i+1}. {sample['prompt'][:100]}{'...' if len(sample['prompt']) > 100 else ''}")
    
    print(f"\nOutput saved to: {args.output_file}")
    print(f"Ready for batch inference with the Qwen2.5-VL script!")


if __name__ == "__main__":
    main()