'''
python3 single_tool_multiround.py \
  --input tool_instruct/*.jsonl \
  --greetings corpus_pack/greetings.txt \
  --banks corpus_pack/phrase_banks.yaml \
  --output multi_round/single_tool_multiround.jsonl

'''

import argparse, json, random, uuid, glob, yaml
from typing import Dict, List, Any, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def dump_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def load_lines(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_banks(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────────────────────
# Enforce alternating human <-> gpt message sequence
# ──────────────────────────────────────────────────────────────────────────────
def enforce_alternation(convo: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_convo = []
    expected = 'human'
    for msg in convo:
        if msg.get('from') == expected:
            new_convo.append(msg)
            expected = 'gpt' if expected == 'human' else 'human'
    # if the last message is from 'human' without a following 'gpt', drop it
    if new_convo and new_convo[-1].get('from') == 'human':
        new_convo.pop()
    return new_convo

# ──────────────────────────────────────────────────────────────────────────────
# Small-talk rounds
# ──────────────────────────────────────────────────────────────────────────────
def chat_round(user_greet: str, agent_reply: str) -> List[Dict[str, Any]]:
    return [
        {'from': 'human', 'value': user_greet},
        {'from': 'gpt',   'thoughts': '…', 'actions': [], 'value': agent_reply},
        {'from': 'human', 'value': "I'm ready to proceed."},
        {'from': 'gpt',   'thoughts': '…', 'actions': [], 'value': 'Sure -- what would you like me to do?'},
    ]

def closing_round(user_small: str, agent_small: str) -> List[Dict[str, Any]]:
    return [
        {'from': 'human', 'value': user_small},
        {'from': 'gpt',   'thoughts': '…', 'actions': [], 'value': agent_small},
        {'from': 'human', 'value': "No, that's all -- thanks!"},
        {'from': 'gpt',   'thoughts': '…', 'actions': [], 'value': 'Glad to help -- take care!'},
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Insert random small-talk throughout original conversation
# ──────────────────────────────────────────────────────────────────────────────
def insert_small_talk(convo: List[Dict[str, Any]], banks: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_convo = []
    if len(convo) <= 1:
        return convo
    num_insert = random.randint(0, 2)
    positions = sorted(random.sample(range(1, len(convo)), num_insert))
    idx = 0
    for i, msg in enumerate(convo):
        new_convo.append(msg)
        if idx < len(positions) and i == positions[idx]:
            new_convo += chat_round(
                random.choice(banks['starters']),
                random.choice(banks['agent_smalltalk'])
            )
            idx += 1
    return new_convo

# ──────────────────────────────────────────────────────────────────────────────
# Build session: using greetings.txt for prefix, banks for diversity, no refine
# ──────────────────────────────────────────────────────────────────────────────
def build_session(
    example: Dict[str, Any],
    greetings: List[str],
    banks: Dict[str, Any],
    probs: Dict[str, float]
) -> Dict[str, Any]:
    convo = example.get('conversations', [])
    # prefix greeting from greetings.txt
    if random.random() < probs.get('prefix_chat', 0.2):
        convo = chat_round(
            random.choice(greetings),
            random.choice(banks['agent_smalltalk'])
        ) + convo
    # insert small-talk within
    convo = insert_small_talk(convo, banks)
    # optional closing small-talk
    if random.random() < probs.get('closing_chat', 0.3):
        convo += closing_round(
            random.choice(banks['user_smalltalk']),
            random.choice(banks['agent_smalltalk'])
        )
    # enforce human-gpt alternation
    convo = enforce_alternation(convo)
    return {
        'session_id': str(uuid.uuid4()),
        'image_id':   example.get('image_id'),
        'image':      example.get('image'),
        'file_name':  example.get('file_name'),
        'conversations': convo
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def cli():
    ap = argparse.ArgumentParser(description='Multi-round dialogue generator (no refine)')
    ap.add_argument(
        '--input', nargs='+', required=True,
        help='One or many *.jsonl files (e.g. tool_instruct/*.jsonl)'
    )
    ap.add_argument(
        '--greetings', required=True,
        help='Text file with greetings for prefix chat'
    )
    ap.add_argument(
        '--banks', required=True,
        help='YAML file with small-talk banks and probabilities'
    )
    ap.add_argument(
        '--output', required=True,
        help='Output JSONL path'
    )
    return ap.parse_args()


def main():
    args = cli()
    single_rounds = []
    for pat in args.input:
        for p in glob.glob(pat):
            single_rounds.extend(load_jsonl(p))
    greetings = load_lines(args.greetings)
    banks = load_banks(args.banks)
    probs = banks.get('probabilities', {'prefix_chat': 0.2, 'closing_chat': 0.3})
    sessions = [
        build_session(ex, greetings, banks, probs)
        for ex in single_rounds
    ]
    dump_jsonl(sessions, args.output)
    print(f"Wrote {len(sessions)} multi-round sessions to {args.output}")

if __name__ == '__main__':
    main()
