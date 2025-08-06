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

# def dump_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
#     """Save list of dictionaries to JSONL file"""
#     with open(path, 'w', encoding='utf-8') as f:
#         for row in data:
#             f.write(json.dumps(row, ensure_ascii=False) + '\n')

def dump_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            # Quick fix: ensure the JSON string doesn't contain literal newlines
            json_str = json.dumps(row, ensure_ascii=False)
            # Replace any literal newlines that somehow got through
            json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
            f.write(json_str + '\n')

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
    # Use greetings file first, fallback to banks if needed
    if greetings:
        greeting = random.choice(greetings)
    else:
        greeting = random.choice(banks.get('starters', ['Hello!']))
        
    return [
        {
            "from": "human", 
            "value": greeting
        },
        {
            "from": "gpt", 
            "thoughts": "The user is greeting me. I should respond warmly and offer assistance.", 
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
            "thoughts": "The user is closing the conversation. I should respond politely and offer future assistance.", 
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
    1. Greeting round (optional, 80% chance)
    2. Original conversation 
    3. Closing round (optional, 70% chance)
    """
    # Get original conversation
    original_convo = example.get('conversations', [])
    
    # Build the complete conversation
    conversation = []
    
    # 1. Optionally add greeting round (80% chance)
    if random.random() < 0.8:
        conversation.extend(create_greeting_round(greetings, banks))
    
    # 2. Add original conversation (single-round dialogue)
    conversation.extend(original_convo)
    
    # 3. Optionally add closing round (70% chance)  
    if random.random() < 0.7:
        conversation.extend(create_closing_round(banks))
    
    # Create session object with all original metadata
    session = {}
    
    # Copy all original fields
    for key, value in example.items():
        if key != 'conversations' and key != "image_id" and key != "file_name":  # Don't copy the old conversations
            session[key] = value
    
    # Add the new conversation
    session['conversations'] = conversation
    
    # # Add session ID if not present
    # if 'session_id' not in session:
    #     session['session_id'] = str(uuid.uuid4())
    
    return session

# ──────────────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-round dialogues by adding greetings/closings to single-round conversations'
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
            data = load_jsonl(file_path)
            single_rounds.extend(data)
            # print(f"    Loaded {len(data)} conversations")
    
    print(f"Total loaded: {len(single_rounds)} single-round conversations")
    
    # Load greetings and phrase banks
    # print("Loading greetings and phrase banks...")

    greetings = load_lines(args.greetings)
    # print(f"Loaded {len(greetings)} greetings")
    
    banks = load_banks(args.banks)
    # print(f"Loaded phrase banks with keys: {list(banks.keys())}")
    
    # Generate multi-round sessions
    print("Generating multi-round sessions...")
    sessions = []
    greeting_count = 0
    closing_count = 0
    
    for i, example in enumerate(single_rounds):
        # if i % 1000 == 0 and i > 0:
            # print(f"  Processed {i}/{len(single_rounds)} conversations...")
            
        session = build_session(example, greetings, banks)
        sessions.append(session)
        
        # Count added greetings/closings for statistics
        conv_len = len(session['conversations'])
        orig_len = len(example.get('conversations', []))
        
        if conv_len > orig_len:
            # Check if greeting was added (first turn is human greeting)
            if session['conversations'][0].get('from') == 'human' and \
               session['conversations'][0].get('value') in greetings + banks.get('starters', []):
                greeting_count += 1
            
            # Check if closing was added (last turn is agent closing)
            if session['conversations'][-1].get('from') == 'gpt' and \
               len(session['conversations']) > orig_len + 2:  # +2 for greeting round
                closing_count += 1
    
    # Save output
    print(f"Saving {len(sessions)} sessions to {args.output}")

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    dump_jsonl(sessions, args.output)
    print("Done!")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Original conversations: {len(single_rounds)}")
    print(f"  Generated sessions: {len(sessions)}")
    print(f"  Added greetings: {greeting_count} ({greeting_count/len(sessions)*100:.1f}%)")
    print(f"  Added closings: {closing_count} ({closing_count/len(sessions)*100:.1f}%)")
    
    # Sample output preview
    if sessions:
        print(f"\nSample conversation structure:")
        sample = sessions[0]
        print(f"  Total turns: {len(sample['conversations'])}")
        print(f"  Fields: {list(sample.keys())}")
        for i, turn in enumerate(sample['conversations'][:3]):  # Show first 3 turns
            from_field = turn.get('from', 'unknown')
            value = turn.get('value', '')[:50] + '...' if len(turn.get('value', '')) > 50 else turn.get('value', '')
            print(f"    Turn {i+1} [{from_field}]: {value}")

if __name__ == '__main__':
    main()
