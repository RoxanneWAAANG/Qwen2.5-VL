import json
import random
import os
from pathlib import Path
from tqdm import tqdm

INPUT_DIR    = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text")
OUTPUT_FILE  = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/llava_rg_dataset.jsonl")

report_generation_instructions = [
    "Generate a medical report based on the findings below.",
    "Write a structured medical report from this clinical note.",
    "Convert the following abstract into a formal report.",
    "Compose an imaging report based on the provided text.",
    "Please create a diagnostic report from the clinical passage.",
    "Draft a medical document describing the case details.",
    "Construct a formal patient report using the following input.",
    "Produce a clinical narrative based on the case summary below.",
    "Generate a detailed report summarizing the diagnostic findings.",
    "Write a structured progress note from this information.",
    "Prepare a discharge summary based on the case description.",
    "Document a radiology-style report for the scenario below.",
    "Create a formatted clinical report for the following patient history.",
    "Write a SOAP-style report based on the provided medical text.",
    "Generate a comprehensive structured report suitable for recordkeeping.",
    "Compose a succinct medical report highlighting key observations.",
    "Convert this clinical abstract into a full diagnostic report.",
    "Write an imaging findings report from the text below.",
    "Produce a patient case report in a standard medical format.",
    "Draft a narrative medical report detailing the findings.",
    "Create a clinical summary report from the given notes.",
    "Generate a formal medical report with headings and sections.",
    "Write a detailed report of the examination findings.",
    "Produce a radiology impression report from this input.",
    "Compose a pathology-style report based on the provided text.",
    "Create a patient consultation report summarizing recommendations.",
    "Write a structured summary report suitable for EMR entry.",
    "Generate a template-driven medical report from the case notes.",
    "Draft a clinician-friendly report interpreting the findings.",
    "Produce a report outlining diagnosis and plan from this summary.",
    "Compose a complete medical report with diagnosis, findings, and plan.",
    "Convert these clinical observations into a diagnostic report.",
    "Write a standardized report for medical record documentation.",
    "Generate a follow-up report based on the patient's history.",
    "Produce an admission report summarizing presenting problems.",
    "Draft a comprehensive health report from the clinical summary.",
    "Create a detailed exam report including impressions and recommendations.",
    "Write a clinical case report in a structured medical format.",
    "Generate an outpatient visit report from the provided data.",
    "Compose a concise report highlighting diagnostic impressions.",
    "Create a professional medical report suitable for peer review.",
    "Write a formatted case study report based on the clinical abstract.",
    "Produce a discharge report covering diagnosis, treatment, and follow-up.",
    "Draft a multidisciplinary medical report summarizing all findings.",
    "Generate an operative report based on the surgical summary.",
    "Compose a report for a tumor board presentation using the case details.",
    "Create a report summarizing imaging, labs, and clinical assessment.",
    "Write a research-style case report from the clinical vignette.",
    "Produce a structured report for quality-assurance review.",
    "Draft a medical report with an emphasis on key learning points."
]

answer_templates = [
    "Here is the completed medical report:\n{report}",
    "Below is the structured report derived from the text:\n{report}",
    "I have generated the formal report as requested:\n{report}",
    "Please review the following diagnostic report:\n{report}",
    "Completed report in standard medical format:\n{report}",
    "The full clinical report is provided here:\n{report}",
    "Here's the comprehensive medical report:\n{report}",
    "Structured report with diagnosis and plan:\n{report}",
    "Detailed report based on the input note:\n{report}",
    "Full narrative report follows:\n{report}",
    "Here is a formatted medical report:\n{report}",
    "The generated report outlining key findings:\n{report}",
    "Complete diagnostic summary:\n{report}",
    "Below find the finalized medical report:\n{report}",
    "Here's the report suitable for EMR entry:\n{report}",
    "Report including impressions and recommendations:\n{report}",
    "Here is the clinician-ready report:\n{report}",
    "Comprehensive structured report:\n{report}",
    "The report has been drafted as follows:\n{report}",
    "Diagnostic report prepared:\n{report}",
    "Full patient report:\n{report}",
    "The following report summarizes all findings:\n{report}",
    "Generated medical report:\n{report}",
    "Structured narrative report:\n{report}",
    "Here is the formal report text:\n{report}",
    "Completed diagnostic report below:\n{report}",
    "Here's the standard formatted report:\n{report}",
    "Report with headings and sections:\n{report}",
    "Finalized report for the clinical passage:\n{report}",
    "Medical report generated:\n{report}",
    "The report is ready for review:\n{report}",
    "Below is the thorough medical report:\n{report}",
    "Comprehensive patient report:\n{report}",
    "Here is the structured diagnostic summary:\n{report}",
    "Reporting completed, see below:\n{report}",
    "Full clinical summary report:\n{report}",
    "Here's the concise yet complete report:\n{report}",
    "Generated narrative report:\n{report}",
    "Report outlining findings and plan:\n{report}",
    "Complete structured document:\n{report}",
]

def load_clinical_texts(folder: Path, max_samples: int) -> list[str]:
    """Load up to max_samples clinical text snippets from .txt files."""
    texts = []
    for fname in sorted(folder.iterdir()):
        if fname.suffix == ".txt":
            content = fname.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
            if len(texts) >= max_samples:
                break
    return texts

def transform(idx: int, text: str) -> dict:
    instruction = random.choice(report_generation_instructions)
    user_prompt = f"{instruction}\n\n{text}"

    tool_call = {
        "from": "gpt",
        "thoughts": "Using LLaVA to generate a formal medical report based on the input.",
        "actions": [
            {
                "API_name": "LLaVA",
                "API_params": {"task": "report_generation", "text": text}
            }
        ],
        "value": "Generating medical report using LLaVA..."
    }

    final_reply = random.choice(answer_templates).format(report="<output_report>")
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"llava_report_{idx}",
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            assistant_reply
        ]
    }

def build_dataset():
    random.seed(42)
    texts = load_clinical_texts(INPUT_DIR, max_samples=5000)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(5000), desc="Building report samples"):
            text = texts[idx % len(texts)]
            record = transform(idx, text)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved samples to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    build_dataset()
