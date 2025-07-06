'''
python3 single_tool_multiround.py \
  --input  tool_instruct/*.jsonl \
  --tool_meta  corpus_pack/tool_meta.yaml \
  --greetings  corpus_pack/greetings.txt \
  --followups  corpus_pack/followups.txt \
  --banks      corpus_pack/phrase_banks.yaml \
  --variables  corpus_pack/variables.yaml \
  --multiplier 1 \
  --output     multi_round/single_tool_multiround.jsonl

'''

import argparse, json, random, uuid, glob, itertools, yaml, pathlib
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
            f.write(json.dumps(row) + "\n")


def load_lines(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def random_vars(var_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """Pick one value for every placeholder key."""
    return {k: random.choice(v) for k, v in var_dict.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Core synthesis logic (a few small functions, no class needed)
# ──────────────────────────────────────────────────────────────────────────────
def tool_and_task(msg):
    actions = msg.get("actions", [])
    if actions:
        tool = actions[0]["API_name"]
    else:
        tool = ""
    return tool, msg.get("task", "")


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


def refine_round(example: Dict[str, Any],
                 tool_meta: Dict[str, Any],
                 followups: List[str],
                 variables: Dict[str, List[str]]) -> List[Dict]:
    tool, _ = tool_and_task(example["conversations"][1])
    meta = tool_meta[tool]
    img_ref = example["image"]

    # follow-up sentence with placeholders filled
    follow = random.choice(followups).format(**random_vars(variables))
    human_req = f"{follow} {meta['refine_task']} on this {meta['modality']} ({img_ref})."

    return [
        {"from": "human",
         "value": human_req},
        {"from": "gpt",
         "thoughts": "…",
         "actions": [{"API_name": tool, "API_params": meta["default_args"]}],
         "value": f"Calling {tool} to {meta['refine_task']}…"},
        {"from": "human",
         "value": f"{tool} output: refined_{img_ref}"},
        {"from": "gpt",
         "thoughts": "…",
         "actions": [],
         "value": random.choice(meta["refine_responses"])},
    ]


def build_session(example: Dict[str, Any],
                  banks: Dict[str, List[str]],
                  tool_meta: Dict[str, Any],
                  followups: List[str],
                  variables: Dict[str, List[str]],
                  probs: Dict[str, float]) -> Dict[str, Any]:

    chat_prefix = random.random() < probs["prefix_chat"]
    chat_closing = random.random() < probs["closing_chat"]

    convo: List[Dict] = []

    # prefix small-talk
    if chat_prefix:
        convo += chat_round(random.choice(banks["starters"]).strip(),
                            random.choice(banks["agent_smalltalk"]))

    # original single-round from source jsonl
    convo += example["conversations"]

    # refine round (mandatory)
    convo += refine_round(example, tool_meta, followups, variables)

    # optional closing small-talk
    if chat_closing:
        convo += closing_round(random.choice(banks["user_smalltalk"]),
                               random.choice(banks["agent_smalltalk"]))

    # wrap session
    return {
        "session_id": str(uuid.uuid4()),
        "image_id":   example["image_id"],
        "image":      example["image"],
        "file_name":  example["file_name"],
        "conversations": convo
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def cli():
    ap = argparse.ArgumentParser(description="Multi-round dialogue generator (simple edition)")
    ap.add_argument("--input", nargs="+", required=True,
                    help="one or many *.jsonl files, wildcards ok (e.g. data/*.jsonl)")
    ap.add_argument("--tool_meta", required=True)
    ap.add_argument("--greetings", required=True)        # txt ➔ starters list
    ap.add_argument("--followups", required=True)        # txt with placeholders
    ap.add_argument("--banks", required=True)            # YAML (transitions, small-talk, completions)
    ap.add_argument("--variables", required=True)        # YAML placeholder dictionaries
    ap.add_argument("--multiplier", type=int, default=1)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def main():
    args = cli()

    # gather single-round examples from all paths
    single_rounds: List[Dict[str, Any]] = []
    for pat in args.input:
        for p in glob.glob(pat):
            single_rounds += load_jsonl(p)

    tool_meta   = yaml.safe_load(open(args.tool_meta))
    followups   = load_lines(args.followups)
    banks       = yaml.safe_load(open(args.banks))
    variables   = yaml.safe_load(open(args.variables))
    probs       = banks.get("probabilities", {"prefix_chat": 0.2, "closing_chat": 0.3})

    sessions: List[Dict[str, Any]] = []
    for ex in single_rounds:
        for _ in range(args.multiplier):
            sessions.append(
                build_session(ex, banks, tool_meta, followups, variables, probs)
            )

    dump_jsonl(sessions, args.output)
    print(f"✔ Wrote {len(sessions)} multi-round sessions to {args.output}")


if __name__ == "__main__":
    main()
