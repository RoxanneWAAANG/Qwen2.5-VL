#!/usr/bin/env python3
"""
Inference script for Qwen2.5-VL with separate checkpoint and processor directories,
including fallback for missing chat_template by loading it from the base model.

Usage:
    python3 ./scripts/infer.py \
      --ckpt_dir /home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/output_7b/checkpoint-121 \
      --processor_dir ./qwen-vl-finetune/output \
      --image_path path/to/image.jpg \
      --prompt "Describe this image."

Prerequisites:
    pip install transformers==4.49.0 accelerate qwen-vl-utils[decord]
    # Optional for FlashAttention2 (faster, if you have compatible hardware):
    # pip install -U flash-attn --no-build-isolation
"""
# export PYTHONPATH=$PYTHONPATH:/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-utils/src

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Base model identifier for chat_template fallback
BASE_MODEL_NAME = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen2.5-VL-3B-Instruct"


def load_model_and_processor(ckpt_dir: str, processor_dir: str):
    """
    Load model weights from ckpt_dir and processor configuration from processor_dir.
    If the loaded processor lacks a chat_template, fetch it from the base model.
    """
    # Load fine-tuned model weights
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load (possibly incomplete) processor configs
    processor = AutoProcessor.from_pretrained(processor_dir)

    # Ensure chat_template is set (needed for apply_chat_template)
    if not hasattr(processor, 'chat_template') or processor.chat_template is None:
        base_proc = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
        processor.chat_template = base_proc.chat_template

    return model, processor


def build_messages(image_path: str, prompt: str, is_video: bool = False):
    """
    Build the chat-style messages payload from image/video and prompt.
    """
    content_item = {"type": "video", "video": image_path} if is_video else {"type": "image", "image": image_path}
    return [
        {"role": "user", "content": [content_item, {"type": "text", "text": prompt}]}  
    ]


def run_inference(model, processor, messages):
    """
    Run the multimodal inference and return decoded text responses.
    """
    # Prepare chat prompt and vision inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize and batch
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate with the model
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Remove prompt tokens from output
    response_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # Decode response
    output = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference with fallback chat_template")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to the checkpoint directory containing model weights")
    parser.add_argument("--processor_dir", type=str, required=True,
                        help="Path to the processor config directory (e.g., output)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to an image or video file for inference")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to accompany the image/video")
    parser.add_argument("--video", action="store_true",
                        help="Flag if the input is a video instead of an image")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.ckpt_dir, args.processor_dir)
    messages = build_messages(args.image_path, args.prompt, is_video=args.video)
    output = run_inference(model, processor, messages)

    print("Model response:")
    print(output[0])


if __name__ == "__main__":
    main()
