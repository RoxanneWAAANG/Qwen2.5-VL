import json
import os
import random
from pathlib import Path
from tqdm import tqdm

INPUT_DIR   = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text")
SUMMARY_DIR = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/abstract")
OUTPUT_FILE = Path("./tool_instruct/llava_sum_dataset.jsonl")

summarization_instructions = [
    "Summarize the key findings of this medical passage.",
    "Generate a concise overview of the medical abstract.",
    "Create a short summary highlighting the main points.",
    "What is the core message in the clinical note below?",
    "Give a brief textual summary of the input content.",
    "Write a summary that captures the essential medical details.",
    "Condense the following clinical passage into a short summary.",
    "What are the main diagnoses or treatments mentioned here?",
    "Briefly summarize the patient case described below.",
    "Provide a one-paragraph summary of the following abstract.",
    "Summarize this clinical report for a busy practitioner.",
    "Generate a synopsis highlighting key symptoms and interventions.",
    "Write a summary suitable for a medical case database.",
    "Create a plain-language summary of this clinical abstract.",
    "Offer a summary of the most critical clinical findings.",
    "Extract and summarize the diagnostic conclusions from this report.",
    "Provide a brief summary emphasizing patient outcomes.",
    "Outline the prominent clinical observations in a short summary.",
    "Reduce the text to a concise summary of major medical insights.",
    "Summarize the methodology and results presented in this excerpt.",
    "Highlight the key clinical recommendations in summary form.",
    "Create a digest of the most important research findings.",
    "Provide an overview suitable for medical record notes.",
    "Write a brief abstract based on the given clinical paragraph.",
    "Generate a short summary focusing on diagnostic criteria.",
    "Summarize the main pharmacological interventions described.",
    "Offer a quick summary for clinician reference.",
    "Provide a concise recap of the clinical trial results.",
    "Create a summary that outlines patient demographics and outcomes.",
    "Summarize this text for inclusion in a discharge summary.",
    "Write a brief summary capturing essential laboratory findings.",
    "Generate a concise overview of symptoms and treatment plans.",
    "Summarize the clinical significance of the findings below.",
    "Offer a one-sentence summary of the passage's main conclusion.",
    "Craft a high-level overview of this patient's journey and outcomes.",
    "Produce a brief summary emphasizing the study's purpose and conclusion.",
    "Capture the main imaging findings in a few sentences.",
    "Summarize the patient's history, exam, and plan in a concise paragraph.",
    "Generate a layperson-friendly summary of this medical text.",
    "Provide a clinical take-home points summary.",
    "Distill this medical abstract into few key sentences.",
    "Summarize the treatment plan and follow-up instructions.",
    "Create a summary that highlights safety and efficacy results.",
    "Produce an executive summary of this clinical trial report.",
    "Write a short summary of the patient's vital signs and labs.",
    "Generate a focused summary on the diagnostic imaging findings.",
    "Summarize the procedural steps and outcomes outlined below.",
    "Provide an abbreviated summary of this medical case.",
    "Create a concise summary for rapid clinical decision-making.",
    "Summarize the adverse events and management strategies.",
    "Write a brief summary of the study design and endpoints.",
    "Produce a summary emphasizing changes from baseline values.",
    "Summarize the follow-up recommendations.",
    "Capture the essential diagnostic criteria in a short summary.",
    "Generate a summary that outlines risk factors and prevention.",
    "Summarize the pathophysiology and key clinical markers.",
    "Provide a summary focusing on patient symptoms and response.",
    "Write a summary of the research hypothesis and results."
]

answer_templates = [
    "Here is the concise summary:\n{summary}",
    "Below is the generated summary:\n{summary}",
    "I have distilled the passage as follows:\n{summary}",
    "Summary of the text:\n{summary}",
    "Here's a brief overview:\n{summary}",
    "The key points are summarized here:\n{summary}",
    "Here is your requested summary:\n{summary}",
    "Completed summary:\n{summary}",
    "Please review the following summary:\n{summary}",
    "Here is the distilled content:\n{summary}",
    "Summary provided below:\n{summary}",
    "Here's the short synopsis:\n{summary}",
    "The main ideas are:\n{summary}",
    "Condensed summary:\n{summary}",
    "This is the brief summary:\n{summary}",
    "Generated concise summary:\n{summary}",
    "The passage has been summarized:\n{summary}",
    "Key findings summarized below:\n{summary}",
    "Summary text:\n{summary}",
    "Here is the core message:\n{summary}",
    "Brief summary follows:\n{summary}",
    "Synopsis of the text:\n{summary}",
    "Essential points:\n{summary}",
    "Overview:\n{summary}",
    "Take-home summary:\n{summary}",
    "Main conclusions:\n{summary}",
    "Executive summary:\n{summary}",
    "Here's the digest:\n{summary}",
    "One-paragraph summary:\n{summary}",
    "Short summary:\n{summary}",
    "Summary in plain language:\n{summary}",
    "Focused summary:\n{summary}",
    "Compact summary:\n{summary}",
    "Here is a quick recap:\n{summary}",
    "Clinical summary:\n{summary}",
    "Summarized content:\n{summary}",
    "Here is the abridged version:\n{summary}",
    "The essence of the text:\n{summary}",
    "Highlighted points:\n{summary}",
    "Concise summary output:\n{summary}",
]

def load_pairs(max_samples: int) -> list[tuple[str, str, str]]:
    summaries = {}
    for f in SUMMARY_DIR.iterdir():
        if f.suffix != ".txt": continue
        key = f.stem.split("_")[-1]
        summaries[key] = f.read_text(encoding="utf-8").strip()

    pairs = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix != ".txt": continue
        key = f.stem.split("_")[-1]
        if key not in summaries: continue
        paragraph = f.read_text(encoding="utf-8").strip()
        if paragraph:
            pairs.append((key, paragraph, summaries[key]))
        if len(pairs) >= max_samples:
            break
    return pairs

def transform(idx: int, key: str, text: str, summary: str) -> dict:
    """Create a single conversation record."""
    instruction = random.choice(summarization_instructions)
    user_prompt = f"{instruction}\n\n{text}"

    tool_call = {
        "from": "gpt",
        "thoughts": "Using LLaVA to summarize the medical passage.",
        "actions": [
            {
                "API_name": "LLaVA",
                "API_params": {"task": "summarization", "text": text}
            }
        ],
        "value": "Generating medical summary with LLaVA..."
    }

    final_reply = random.choice(answer_templates).format(summary=summary)
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"summ_{key}_{idx}",
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            assistant_reply
        ]
    }

def build_dataset(n_samples: int = 5000,
                  seed: int = 42,
                  output_path: Path = OUTPUT_FILE) -> None:
    random.seed(seed)
    pairs = load_pairs(n_samples)
    print(f"Loaded {len(pairs)} paragraph-summary pairs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, (key, para, summ) in enumerate(tqdm(pairs, desc="Building summaries")):
            json.dump(transform(idx, key, para, summ),
                      fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {len(pairs)} summarization samples to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
