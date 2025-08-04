'''
python3 single_tool_multiround.py \
  --input tool_instruct/*.jsonl \
  --greetings corpus_pack/greetings.txt \
  --banks corpus_pack/phrase_banks.yaml \
  --output multi_round/single_tool_multiround.jsonl
'''

import argparse
import json
import random
import uuid
import glob
import yaml
from typing import Dict, List, Any, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def dump_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
    """Save list of dictionaries to JSONL file"""
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def load_lines(path: str) -> List[str]:
    """Load text file into list of lines"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_banks(path: str) -> Dict[str, Any]:
    """Load YAML phrase banks"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────────────────────
# Conversation building
# ──────────────────────────────────────────────────────────────────────────────
def create_greeting_round(greetings: List[str], banks: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create opening greeting round"""
    # Use starters from banks if greetings file is empty, otherwise use greetings file
    if not greetings:
        greeting = random.choice(banks.get('starters', ['Hello! ']))
    else:
        greeting = random.choice(greetings)
        
    return [
        {
            "from": "human", 
            "value": greeting
        },
        {
            "from": "gpt", 
            "thoughts": "…", 
            "actions": [], 
            "value": random.choice(banks.get('agent_smalltalk', ['Hello! How can I help you today?']))
        }
    ]

def create_closing_round(banks: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create closing round with random phrases"""
    # Use dedicated closing phrases if available, otherwise fall back to smalltalk
    user_closing_phrases = banks.get('user_closing', banks.get('user_smalltalk', ["That's all for now, thanks!"]))
    agent_closing_phrases = banks.get('agent_closing', banks.get('agent_smalltalk', ["Glad I could help! Have a great day."]))
    
    return [
        {
            "from": "human", 
            "value": random.choice(user_closing_phrases)
        },
        {
            "from": "gpt", 
            "thoughts": "…", 
            "actions": [], 
            "value": random.choice(agent_closing_phrases)
        }
    ]

def build_session(
    example: Dict[str, Any],
    greetings: List[str],
    banks: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a complete session with:
    1. Greeting round
    2. Original conversation 
    3. Closing round
    """
    # Get original conversation
    original_convo = example.get('conversations', [])
    
    # Build the complete conversation
    conversation = []
    
    # 1. Add greeting round
    conversation.extend(create_greeting_round(greetings, banks))
    
    # 2. Add original conversation (single-round dialogue)
    conversation.extend(original_convo)
    
    # 3. Add closing round
    conversation.extend(create_closing_round(banks))
    
    # Create session object
    session = {
        'session_id': str(uuid.uuid4()),
        'conversations': conversation
    }
    
    # Preserve original metadata if present
    if 'image_id' in example:
        session['image_id'] = example['image_id']
    if 'image' in example:
        session['image'] = example['image']
    if 'file_name' in example:
        session['file_name'] = example['file_name']
    
    return session

# ──────────────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate simplified multi-round dialogues'
    )
    parser.add_argument(
        '--input', 
        nargs='+', 
        required=True,
        help='Input JSONL file(s) with single-round conversations'
    )
    parser.add_argument(
        '--greetings', 
        required=True,
        help='Text file containing greeting phrases'
    )
    parser.add_argument(
        '--banks', 
        required=True,
        help='YAML file with phrase banks for agent responses'
    )
    parser.add_argument(
        '--output', 
        required=True,
        help='Output JSONL file path'
    )
    
    args = parser.parse_args()
    
    # Load all input files
    print("Loading input files...")
    single_rounds = []
    for pattern in args.input:
        for file_path in glob.glob(pattern):
            print(f"  Loading: {file_path}")
            single_rounds.extend(load_jsonl(file_path))
    
    print(f"Loaded {len(single_rounds)} single-round conversations")
    
    # Load greetings and phrase banks
    print("Loading greetings and phrase banks...")
    greetings = load_lines(args.greetings)
    banks = load_banks(args.banks)
    
    print(f"Loaded {len(greetings)} greetings")
    
    # Generate multi-round sessions
    print("Generating multi-round sessions...")
    sessions = []
    for example in single_rounds:
        session = build_session(example, greetings, banks)
        sessions.append(session)
    
    # Save output
    print(f"Saving {len(sessions)} sessions to {args.output}")
    dump_jsonl(sessions, args.output)
    
    print("Done!")

if __name__ == '__main__':
    main()
