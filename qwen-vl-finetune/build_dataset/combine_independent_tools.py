#!/usr/bin/env python3
"""
combine_multi_tool_conversations.py

Combine multi-turn tool conversations from different tool-specific JSONL files 
into multi-round sessions where each session includes 2-3 independent tools.

Usage:
    python3 combine_independent_tools.py \
        --inputs tool_instruct/healthgpt_reconstruct_dataset.jsonl tool_instruct/healthgpt_superres_dataset.jsonl tool_instruct/internet_seg_dataset.jsonl tool_instruct/llava_rad_rg_dataset.jsonl tool_instruct/llava_sum_dataset.jsonl tool_instruct/pmc_llama_medqa_dataset.jsonl tool_instruct/rate_ner_dataset.jsonl tool_instruct/svlms_fundus_dataset.jsonl tool_instruct/ultrasam_seg_dataset.jsonl tool_instruct/unigradicon_reg_dataset.jsonl \
        --output multi_tool_sessions.jsonl \
        --num_sessions 50 \
        --min_tools 2 \
        --max_tools 3 \
        [--seed 42]
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any

def load_tool_conversations(input_paths: List[str]) -> Dict[str, List[Dict]]:
    """
    Load conversations from tool-specific JSONL files.
    Returns a dict mapping tool_name -> list of conversations.
    """
    tool_conversations = {}
    
    for path in input_paths:
        tool_name = os.path.splitext(os.path.basename(path))[0]
        conversations = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    conversations.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON in {path}:{line_num}: {e}")
        
        tool_conversations[tool_name] = conversations
        print(f"Loaded {len(conversations)} conversations from {tool_name}")
    
    return tool_conversations

def convert_conversation_to_qwen_format(conversation_data: Dict) -> List[Dict]:
    """
    Convert a single tool conversation to Qwen ChatML format.
    Returns list of messages.
    """
    messages = []
    conversations = conversation_data.get("conversations", [])
    image_path = conversation_data.get("file_name", "")
    
    for i, turn in enumerate(conversations):
        role = "human" if turn["from"] == "human" else "gpt"
        content_text = turn["value"]
        
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": content_text.replace("<image>\n", "").strip()}
        ]
        
        messages.append({
            "role": role,
            "content": content
        })
    
    return messages

def create_multi_tool_session(selected_conversations: List[Dict], tool_names: List[str]) -> Dict:
    """
    Create a multi-tool session by chaining conversations from different tools.
    """
    all_messages = []
    
    # Add system message
    system_msg = {
        "role": "system", 
        "content": f"You are a medical AI assistant with access to specialized tools: {', '.join(tool_names)}. You can help users with various tasks using these capabilities."
    }
    all_messages.append(system_msg)
    
    # Add conversations from each tool
    for conv_data in selected_conversations:
        tool_messages = convert_conversation_to_qwen_format(conv_data)
        
        # Skip system messages from individual conversations if they exist
        for msg in tool_messages:
            if msg["role"] != "system":
                all_messages.append(msg)
    
    return {"messages": all_messages}

def sample_diverse_conversations(tool_conversations: Dict[str, List[Dict]], 
                               num_tools: int) -> tuple:
    """
    Sample one conversation from each of the specified number of different tools.
    Returns (selected_conversations, tool_names).
    """
    available_tools = list(tool_conversations.keys())
    tool_name_mapping = {
        'healthgpt_reconstruct_dataset': 'HealthGPT',
        'healthgpt_superres_dataset': 'HealthGPT',
        'internet_seg_dataset': 'IterNet',
        'llava_rad_rg_dataset': 'LLaVA-Rad',
        'pmc_llama_medqa_dataset': 'PMC-LLaMA',
        'llava_sum_dataset': 'LLaVA',
        'rate_ner_dataset': 'RaTE-NER',
        'svlms_fundus_dataset': 'SpecialistVLMs',
        'ultrasam_seg_dataset': 'UltraSAM',
        'unigradicon_reg_dataset': 'UniGradICON'
    }
    
    if len(available_tools) < num_tools:
        raise ValueError(f"Not enough tools available. Need {num_tools}, have {len(available_tools)}")
    
    # Randomly select tools
    selected_tools = random.sample(available_tools, num_tools)
    
    # Sample one conversation from each selected tool
    selected_conversations = []
    for tool in selected_tools:
        if not tool_conversations[tool]:
            raise ValueError(f"No conversations available for tool {tool}")
        conv = random.choice(tool_conversations[tool])
        selected_conversations.append(conv)
    
    # Map tool names to their display names
    selected_tools = [tool_name_mapping.get(tool, tool) for tool in selected_tools]
    
    return selected_conversations, selected_tools

def combine_multi_tool_conversations(input_paths: List[str], 
                                   output_path: str, 
                                   num_sessions: int,
                                   min_tools: int, 
                                   max_tools: int, 
                                   seed: int = None):
    """
    Main function to combine multi-tool conversations.
    """
    if seed is not None:
        random.seed(seed)
    
    # Load all tool conversations
    tool_conversations = load_tool_conversations(input_paths)
    
    if len(tool_conversations) < min_tools:
        raise RuntimeError(f"Need at least {min_tools} different tools, but only found {len(tool_conversations)}")
    
    sessions = []
    
    for session_idx in range(num_sessions):
        try:
            # Randomly decide how many tools to use in this session
            num_tools = random.randint(min_tools, max_tools)
            num_tools = min(num_tools, len(tool_conversations))  # Don't exceed available tools
            
            # Sample conversations from different tools
            selected_conversations, selected_tools = sample_diverse_conversations(
                tool_conversations, num_tools
            )
            
            # Create the multi-tool session
            session = create_multi_tool_session(selected_conversations, selected_tools)
            sessions.append(session)
            
            if (session_idx + 1) % 1000 == 0:
                print(f"Generated {session_idx + 1}/{num_sessions} sessions...")
                
        except Exception as e:
            print(f"Error creating session {session_idx}: {e}")
            continue
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for session in sessions:
            f.write(json.dumps(session, ensure_ascii=False) + '\n')
    
    print(f"Successfully generated {len(sessions)} multi-tool sessions")
    return len(sessions)

def main():
    parser = argparse.ArgumentParser(
        description="Combine multi-turn tool conversations into multi-tool sessions compatible with Qwen2.5-VL."
    )
    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="List of input JSONL files, each containing conversations for a specific tool."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSONL file for multi-tool sessions."
    )
    parser.add_argument(
        "--num_sessions", "-n",
        type=int,
        default=10000,
        help="Number of sessions to generate."
    )
    parser.add_argument(
        "--min_tools",
        type=int,
        default=2,
        help="Minimum number of tools per session."
    )
    parser.add_argument(
        "--max_tools",
        type=int,
        default=3,
        help="Maximum number of tools per session."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    
    try:
        actual_sessions = combine_multi_tool_conversations(
            args.inputs,
            args.output,
            args.num_sessions,
            args.min_tools,
            args.max_tools,
            seed=args.seed
        )
        print(f"Generated {actual_sessions} sessions in {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())