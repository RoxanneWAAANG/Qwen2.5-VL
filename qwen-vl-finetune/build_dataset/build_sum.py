import json
import os
import random
from pathlib import Path
from tqdm import tqdm

INPUT_DIR   = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text")
SUMMARY_DIR = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/abstract")
IMAGE_BASE_PATH = "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/dummy_images"
DUMMY_IMAGE_NAME = "dummy_img.png"
OUTPUT_FILE = Path("./tool_instruct/llava_sum_dataset.jsonl")

summarization_instructions = [
    "<image>\nSummarize the key findings of this medical passage.",
    "<image>\nGenerate a concise overview of the medical abstract.",
    "<image>\nCreate a short summary highlighting the main points.",
    "<image>\nWhat is the core message in the clinical note below?",
    "<image>\nGive a brief textual summary of the input content.",
    "<image>\nWrite a summary that captures the essential medical details.",
    "<image>\nCondense the following clinical passage into a short summary.",
    "<image>\nWhat are the main diagnoses or treatments mentioned here?",
    "<image>\nBriefly summarize the patient case described below.",
    "<image>\nProvide a one-paragraph summary of the following abstract.",
    "<image>\nSummarize this clinical report for a busy practitioner.",
    "<image>\nGenerate a synopsis highlighting key symptoms and interventions.",
    "<image>\nWrite a summary suitable for a medical case database.",
    "<image>\nCreate a plain-language summary of this clinical abstract.",
    "<image>\nOffer a summary of the most critical clinical findings.",
    "<image>\nExtract and summarize the diagnostic conclusions from this report.",
    "<image>\nProvide a brief summary emphasizing patient outcomes.",
    "<image>\nOutline the prominent clinical observations in a short summary.",
    "<image>\nReduce the text to a concise summary of major medical insights.",
    "<image>\nSummarize the methodology and results presented in this excerpt.",
    "<image>\nHighlight the key clinical recommendations in summary form.",
    "<image>\nCreate a digest of the most important research findings.",
    "<image>\nProvide an overview suitable for medical record notes.",
    "<image>\nWrite a brief abstract based on the given clinical paragraph.",
    "<image>\nGenerate a short summary focusing on diagnostic criteria.",
    "<image>\nSummarize the main pharmacological interventions described.",
    "<image>\nOffer a quick summary for clinician reference.",
    "<image>\nProvide a concise recap of the clinical trial results.",
    "<image>\nCreate a summary that outlines patient demographics and outcomes.",
    "<image>\nSummarize this text for inclusion in a discharge summary.",
    "<image>\nWrite a brief summary capturing essential laboratory findings.",
    "<image>\nGenerate a concise overview of symptoms and treatment plans.",
    "<image>\nSummarize the clinical significance of the findings below.",
    "<image>\nOffer a one-sentence summary of the passage's main conclusion.",
    "<image>\nCraft a high-level overview of this patient's journey and outcomes.",
    "<image>\nProduce a brief summary emphasizing the study's purpose and conclusion.",
    "<image>\nCapture the main imaging findings in a few sentences.",
    "<image>\nSummarize the patient's history, exam, and plan in a concise paragraph.",
    "<image>\nGenerate a layperson-friendly summary of this medical text.",
    "<image>\nProvide a clinical take-home points summary.",
    "<image>\nDistill this medical abstract into few key sentences.",
    "<image>\nSummarize the treatment plan and follow-up instructions.",
    "<image>\nCreate a summary that highlights safety and efficacy results.",
    "<image>\nProduce an executive summary of this clinical trial report.",
    "<image>\nWrite a short summary of the patient's vital signs and labs.",
    "<image>\nGenerate a focused summary on the diagnostic imaging findings.",
    "<image>\nSummarize the procedural steps and outcomes outlined below.",
    "<image>\nProvide an abbreviated summary of this medical case.",
    "<image>\nCreate a concise summary for rapid clinical decision-making.",
    "<image>\nSummarize the adverse events and management strategies.",
    "<image>\nWrite a brief summary of the study design and endpoints.",
    "<image>\nProduce a summary emphasizing changes from baseline values.",
    "<image>\nSummarize the follow-up recommendations.",
    "<image>\nCapture the essential diagnostic criteria in a short summary.",
    "<image>\nGenerate a summary that outlines risk factors and prevention.",
    "<image>\nSummarize the pathophysiology and key clinical markers.",
    "<image>\nProvide a summary focusing on patient symptoms and response.",
    "<image>\nWrite a summary of the research hypothesis and results."
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

def generate_image_id():
    """Generate a medical image ID for format compatibility
    
    Even though we're using dummy images, we maintain realistic ID format
    to ensure compatibility with systems expecting medical image identifiers
    """
    segments = []
    for _ in range(5):
        segment = format(random.randint(0, 0xffffffff), '08x')
        segments.append(segment)
    return '-'.join(segments)

def create_image_filename(image_id):
    """Create dummy image filename and full path
    
    Uses a single dummy image for all entries since this is a text-based summarization task.
    The image_id is still unique for each record to maintain proper format structure.
    """
    full_path = f"{IMAGE_BASE_PATH}/{DUMMY_IMAGE_NAME}"
    return DUMMY_IMAGE_NAME, full_path

def load_pairs(max_samples: int) -> list[tuple[str, str, str]]:
    """Load text-summary pairs from the input directories
    
    This function maintains the original pairing logic while preparing data
    for the new multimodal format transformation.
    """
    summaries = {}
    for f in SUMMARY_DIR.iterdir():
        if f.suffix != ".txt": 
            continue
        key = f.stem.split("_")[-1]
        summaries[key] = f.read_text(encoding="utf-8").strip()

    pairs = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix != ".txt": 
            continue
        key = f.stem.split("_")[-1]
        if key not in summaries: 
            continue
        
        paragraph = f.read_text(encoding="utf-8").strip()
        if paragraph:
            pairs.append((key, paragraph, summaries[key]))
        
        if len(pairs) >= max_samples:
            break 
    
    return pairs

def transform(idx: int, key: str, text: str, summary: str) -> dict:
    """Transform a text-summary pair into the new multimodal format
    
    This function converts the original summarization data into the four-turn
    conversation pattern while preserving the LLaVA tool integration and 
    medical summarization task structure.
    """
    image_id = generate_image_id()
    image_filename, full_path = create_image_filename(image_id)
    
    instruction = random.choice(summarization_instructions)
    user_prompt = f"{instruction}\n\n### Text to summarize:\n{text}"
    
    llava_output = f"LLaVA output: {summary}\n\nAnswer my first request: {instruction.replace('<image>', '').strip()}\n\n"
    
    final_reply = random.choice(answer_templates).format(summary=summary)
    
    return {
        "image_id": image_id,
        "image": image_filename,
        "file_name": full_path,
        "conversations": [
            {
                "from": "human",
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": "To generate a medical summary of this text, I'll use the LLaVA model.",
                "actions": [
                    {
                        "API_name": "LLaVA",
                        "API_params": {"task": "summarization", "text": text}
                    }
                ],
                "value": "Calling LLaVA to generate a medical summary..."
            },
            {
                "from": "human",
                "value": llava_output
            },
            {
                "from": "gpt",
                "thoughts": "The LLaVA tool has completed the summarization. Now I can provide the formatted summary.",
                "actions": [],
                "value": final_reply
            }
        ]
    }

def build_dataset(n_samples: int = 5000,
                  seed: int = 42,
                  output_path: Path = OUTPUT_FILE) -> None:
    """Build the medical summarization dataset in the new multimodal format
    
    This function orchestrates the entire transformation process while maintaining
    the original data loading and processing logic adapted for multimodal compatibility.
    """
    random.seed(seed)
    
    pairs = load_pairs(n_samples)
    print(f"Loaded {len(pairs)} paragraph-summary pairs for transformation.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, (key, para, summ) in enumerate(tqdm(pairs, desc="Building multimodal summarization dataset")):
            record = transform(idx, key, para, summ)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
    
    print(f"\nSaved {len(pairs)} summarization samples to '{output_path}'")
    print(f"All records use dummy image: {DUMMY_IMAGE_NAME}")

if __name__ == "__main__":
    build_dataset()
