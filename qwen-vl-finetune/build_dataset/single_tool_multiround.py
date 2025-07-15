'''
python3 single_tool_multiround.py \
  --input tool_instruct/*.jsonl \
  --tool_meta corpus_pack/tool_meta.yaml \
  --greetings corpus_pack/greetings.txt \
  --followups corpus_pack/followups.txt \
  --banks corpus_pack/phrase_banks.yaml \
  --variables corpus_pack/variables.yaml \
  --output multi_round/single_tool_multiround.jsonl
'''

import argparse, json, random, uuid, glob, yaml
from typing import Dict, List, Any, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f]

def dump_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_lines(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def random_vars(var_dict: Dict[str, List[str]]) -> Dict[str, str]:
    return {k: random.choice(v) for k, v in var_dict.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Core synthesis logic
# ──────────────────────────────────────────────────────────────────────────────
def tool_and_task(msg):
    actions = msg.get("actions", [])
    tool = actions[0]["API_name"] if actions else ""
    return tool, msg.get("task", "")

# small-talk rounds
def chat_round(user_greet: str, agent_reply: str) -> List[Dict]:
    return [
        {"from": "human", "value": user_greet},
        {"from": "gpt",   "thoughts": "…", "actions": [], "value": agent_reply},
        {"from": "human", "value": "I'm ready to proceed."},
        {"from": "gpt",   "thoughts": "…", "actions": [], "value": "Sure -- what would you like me to do?"},
    ]

def closing_round(user_small: str, agent_small: str) -> List[Dict]:
    return [
        {"from": "human", "value": user_small},
        {"from": "gpt",   "thoughts": "…", "actions": [], "value": agent_small},
        {"from": "human", "value": "No, that's all -- thanks!"},
        {"from": "gpt",   "thoughts": "…", "actions": [], "value": "Glad to help -- take care!"},
    ]

def refine_round(example: Dict[str, Any], tool_meta: Dict[str, Any], followups: List[str], variables: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    tool, _ = tool_and_task(example["conversations"][1])
    meta = tool_meta.get(tool, {})
    img_ref = example.get("image", "<img>")
    follow = random.choice(followups).format(**random_vars(variables))
    human_req = f"{follow} {meta.get('refine_task','refine')} on this {meta.get('modality','item')} ({img_ref})."
    return [
        {"from": "human", "value": human_req},
        {"from": "gpt",   "thoughts": "…", "actions": [{"API_name": tool, "API_params": meta.get("default_args",{})}],
         "value": f"Calling {tool} to {meta.get('refine_task','refine')}…"},
        {"from": "human", "value": f"{tool} output: refined_{img_ref}"},
        {"from": "gpt",   "thoughts": "…", "actions": [],
         "value": random.choice(meta.get("refine_responses", ["Done."]))},
    ]

# insert random small-talk throughout the conversation
def insert_small_talk(convo: List[Dict[str, Any]], banks: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_convo = []
    # decide number of insertions (0 to 2)
    num_insert = random.randint(0, 2)
    positions = sorted(random.sample(range(1, len(convo)), num_insert))
    idx = 0
    for i, msg in enumerate(convo):
        new_convo.append(msg)
        if idx < len(positions) and i == positions[idx]:
            # insert a random small-talk segment
            new_convo += chat_round(random.choice(banks['starters']), random.choice(banks['agent_smalltalk']))
            idx += 1
    return new_convo


def build_session(example: Dict[str, Any], banks: Dict[str, Any], tool_meta: Dict[str, Any], followups: List[str], variables: Dict[str, List[str]], probs: Dict[str, float]) -> Dict[str, Any]:
    convo = example.get("conversations", [])
    # prepend optional small-talk
    if random.random() < probs.get("prefix_chat", 0.2):
        convo = chat_round(random.choice(banks['starters']), random.choice(banks['agent_smalltalk'])) + convo
    # insert small-talk throughout
    convo = insert_small_talk(convo, banks)
    # mandatory refine
    convo += refine_round(example, tool_meta, followups, variables)
    # optional closing
    if random.random() < probs.get("closing_chat", 0.3):
        convo += closing_round(random.choice(banks['user_smalltalk']), random.choice(banks['agent_smalltalk']))
    return {
        "session_id": str(uuid.uuid4()),
        "image_id":   example.get("image_id"),
        "image":      example.get("image"),
        "file_name":  example.get("file_name"),
        "conversations": convo
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def cli():
    ap = argparse.ArgumentParser(description="Multi-round dialogue generator (expanded)")
    ap.add_argument("--input", nargs="+", required=True, help="One or many *.jsonl files")
    ap.add_argument("--tool_meta", required=True)
    ap.add_argument("--greetings", required=True)
    ap.add_argument("--followups", required=True)
    ap.add_argument("--banks", required=True)
    ap.add_argument("--variables", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def main():
    args = cli()
    single_rounds = []
    for pat in args.input:
        for p in glob.glob(pat):
            single_rounds.extend(load_jsonl(p))
    banks     = yaml.safe_load(open(args.banks))
    tool_meta = yaml.safe_load(open(args.tool_meta))
    followups = load_lines(args.followups)
    variables = yaml.safe_load(open(args.variables))
    probs     = banks.get("probabilities", {"prefix_chat":0.2, "closing_chat":0.3})

    sessions = []
    for ex in single_rounds:
        sessions.append(build_session(ex, banks, tool_meta, followups, variables, probs))
    dump_jsonl(sessions, args.output)
    print(f"✔ Wrote {len(sessions)} multi-round sessions to {args.output}")

if __name__ == "__main__":
    main()
