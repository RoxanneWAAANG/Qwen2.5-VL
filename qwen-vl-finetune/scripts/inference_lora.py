#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def load_model_and_adapter(base_model_path, adapter_path, device="auto"):
    """
    Load base model and LoRA adapter
    
    Args:
        base_model_path (str): Path to base model
        adapter_path (str): Path to LoRA adapter
        device (str): Device to load model on
        
    Returns:
        model, processor
    """
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    # Load base model
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    # Merge adapter weights for inference (optional, for better performance)
    model = model.merge_and_unload()
    
    return model, processor


def inference(model, processor, image_path, query, max_new_tokens=128):
    """
    Run inference on image and query
    
    Args:
        model: Loaded model
        processor: Loaded processor  
        image_path (str): Path to image
        query (str): Text query
        max_new_tokens (int): Maximum tokens to generate
        
    Returns:
        Generated text
    """
    # Prepare messages
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": query},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
    
    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LoRA Inference")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, 
                        help="Path to LoRA adapter")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--query", type=str, required=True,
                        help="Text query for the image")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Load model and adapter
    print("Loading model and LoRA adapter...")
    model, processor = load_model_and_adapter(
        args.base_model_path, 
        args.adapter_path,
        args.device
    )
    
    # Run inference
    print(f"Running inference on: {args.image_path}")
    print(f"Query: {args.query}")
    
    output = inference(
        model, 
        processor, 
        args.image_path, 
        args.query,
        args.max_new_tokens
    )
    
    print(f"Response: {output}")


if __name__ == "__main__":
    main()