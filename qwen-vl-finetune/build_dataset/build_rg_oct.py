import json
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR    = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/deepeyenet/deepeyenet")
INPUT_FILE  = BASE_DIR / "DeepEyeNet_train.json"
OUTPUT_FILE = Path("./tool_instruct/svlms_fundus_dataset.jsonl")

instruction_templates = [
    "Can you write a clinical report based on this OCT scan?",
    "Please review the OCT image and summarize your findings.",
    "What does this OCT scan show? Generate a full report.",
    "Take a look at this OCT image and draft a diagnostic summary.",
    "Generate a report for the following OCT image.",
    "Summarize clinical findings from this OCT scan.",
    "Write OCT findings for the attached OCT image.",
    "Create a report: OCT scan, full-field evaluation.",
    "Complete the diagnostic note for this OCT exam.",
    "Describe what you see in the OCT image.",
    "Report all normal and abnormal findings in this OCT scan.",
    "Summarize this OCT image into a clinical report.",
    "Look at the OCT scan and list all key observations.",
    "Give a medical interpretation of this OCT image.",
    "Generate a detailed report based on the image.",
    "Review the image and provide a comprehensive diagnostic report.",
    "Interpret this image and summarize all significant findings.",
    "Write a complete report including impressions and recommendations.",
    "Provide an expert diagnostic report based on the following image.",
    "Evaluate the image and draft a structured report.",
    "You are a assisting a physician. Analyze this image and report your findings.",
    "Compose a diagnostic report covering anatomy, pathology, and relevant observations.",
    "Examine the image and describe any notable abnormalities or variations.",
    "Create a narrative report from the attached scan.",
    "Write a retina specialist's report based on the clinical scan provided.",
    "Review the attached scan and produce a professional report.",
    "Create a structured report summarizing the key radiological findings.",
    "Interpret the scan below and generate a report suitable for clinical review.",
    "Write a full report as if preparing for inclusion in a patient chart.",
    "Could you please review this image and generate a report?",
    "Can you interpret the attached image and summarize your findings?",
    "Would you mind providing a report for this image?",
    "Can you help by writing a diagnostic report based on this scan?",
    "Could you describe what this image shows?",
    "Write a professional report followed by a simple summary a patient can understand.",
    "Generate a OCT scan report with a lay explanation at the end.",
    "Provide a diagnostic report and a brief patient-friendly translation.",
    "Interpret this OCT scan and explain the results in everyday language.",
    "Would you be able to draft a full clinical report for this image?",
    "Can you generate a professional summary from this image?",
    "Could you analyze this scan and outline the major findings?",
    "Would you provide your interpretation of this image?",
    "Could you offer a comprehensive report based on the attached image?",
    "Please provide a structured diagnostic report for this scan.",
    "Can you prepare a full read of the scan results in report format?",
    "What do you see in this image?",
    "Is there anything abnormal in the scan?",
    "What are your observations from this image?",
    "Can you tell what might be wrong from this scan?",
    "Does anything stand out in this image?",
    "How would you interpret this scan?",
    "What are the key features seen in the image?",
    "What is your diagnostic impression of this image?",
    "What findings should be noted in this scan?",
    "Is this image normal or are there signs of OCT?",
    "What can you infer from this scan?",
    "Can you walk me through your radiological interpretation?",
    "What should a radiologist say about this scan?",
    "Do you notice anything concerning in this image?",
    "What might this scan indicate clinically?",
    "You are acting as a clinical radiologist — report on this image.",
    "You're helping with a case review. Please write the findings.",
    "You're a senior retina specialist guiding a trainee — show them how you'd report this scan.",
    "Pretend you are documenting for a hospital EMR system — report this scan.",
    "Generate a diagnostic summary to support a referring physician.",
    "You're reviewing this scan for surgical planning — provide your full read.",
    "Simulate an attending radiologist dictating a clinical report.",
    "Write a report as if it will be reviewed in a multidisciplinary team meeting.",
    "Please report the attached image.",
    "This scan was flagged by a junior — provide your expert interpretation.",
    "As a consulting radiologist, provide a second opinion report for this scan.",
    "What does this OCT image show?",
    "Are there any abnormalities visible in this OCT scan?",
    "What's your impression of this OCT image?",
    "Is anything concerning visible in the following OCT scan?",
    "Can you walk me through your findings from this OCT image?",
    "Do you notice anything unusual in the attached OCT scan?",
    "What are the key observations in this OCT image?",
    "How would you interpret this OCT scan?",
    "What can be concluded from this OCT image?",
    "What findings would you report from the provided OCT scan?",
    "What stands out to you in this OCT image?",
    "Can you detect any critical findings in this OCT scan?",
    "Do you see anything that needs follow-up in the OCT scan?",
    "What is your diagnostic impression from the OCT image?",
    "Does this OCT scan suggest any pathology?",
    "Could this OCT image be interpreted as normal or abnormal?",
    "Could you please generate a report for this OCT image?",
    "Can you provide a detailed interpretation of the OCT scan below?",
    "Would you mind writing a report based on this OCT image?",
    "Could you examine the OCT image and describe your findings?",
    "Can you summarize the diagnostic findings from the OCT image?",
    "Could you generate a structured report for the attached OCT scan?",
    "Would you please interpret this OCT image and report your impressions?",
    "Can you analyze the following OCT scan and share a report?",
    "Could you write a complete diagnostic report based on this OCT image?",
    "Would you be able to produce a report for the attached OCT scan?",
    "Can you create a summary for this OCT scan?",
    "May I ask you to provide a clinical report for this OCT image?",
    "Could you kindly review the OCT scan and share your thoughts?",
    "Would you assist in interpreting the following OCT image?",
    "Please could you generate a comprehensive report for the OCT scan?",
]

answer_templates = [
    "Here is the comprehensive OCT report:\n{report}",
    "Below is the detailed OCT interpretation:\n{report}",
    "I have completed the OCT analysis. Report follows:\n{report}",
    "Diagnostic OCT report:\n{report}",
    "Here's the structured OCT report with key findings:\n{report}",
    "The OCT scan has been reviewed. Full report:\n{report}",
    "Please see the narrative OCT report below:\n{report}",
    "Final OCT assessment:\n{report}",
    "Full ophthalmic report for this OCT scan:\n{report}",
    "Below is the in-depth OCT readout:\n{report}",
    "Comprehensive findings for this OCT image:\n{report}",
    "Here's the organized OCT report:\n{report}",
    "My complete OCT interpretation is as follows:\n{report}",
    "OCT report with impressions and recommendations:\n{report}",
    "Detailed OCT findings:\n{report}",
    "Attached is the finalized OCT report:\n{report}",
    "Here is the diagnostic note for the OCT scan:\n{report}",
    "Full text of the OCT report:\n{report}",
    "Complete OCT summary:\n{report}",
    "Ophthalmic report (OCT):\n{report}",
    "Below find the OCT evaluation:\n{report}",
    "Here is the clinical OCT report:\n{report}",
    "Comprehensive OCT findings and impressions:\n{report}",
    "The OCT analysis is complete. Report:\n{report}",
    "Clinical report for the OCT image:\n{report}",
    "The following OCT report outlines all observations:\n{report}",
    "My OCT read and conclusions:\n{report}",
    "OCT diagnostic summary:\n{report}",
    "Analytical OCT report:\n{report}",
    "Here's the final OCT diagnostic report:\n{report}",
    "Full OCT commentary:\n{report}",
    "Structured OCT assessment:\n{report}",
    "See the OCT report below:\n{report}",
    "The OCT image has been interpreted. Findings:\n{report}",
    "Report generated for the OCT scan:\n{report}",
    "Below is the detailed assessment of the OCT:\n{report}",
    "Here's the completed OCT evaluation:\n{report}",
    "OCT interpretation completed. Report text:\n{report}",
    "Detailed review of OCT image:\n{report}",
    "Complete narrative for this OCT scan:\n{report}",
    "Here is the full OCT diagnostic note:\n{report}",
]

def load_deepeyenet() -> list:
    """Load DeepEyeNet JSON and convert to list of dicts with image_path / report."""
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        rel_path, meta = next(iter(entry.items()))
        full_path = str(BASE_DIR / rel_path)
        report_text = (meta.get("clinical-description") or
                       meta.get("keywords") or "").strip()

        records.append({
            "image_path": [full_path],
            "report": report_text
        })
    return records

def transform(record: dict, idx: int) -> dict:
    user_prompt = instruction_templates[idx % len(instruction_templates)].format(
        modality='OCT'
    )

    tool_call = {
        "from": "gpt",
        "thoughts": "User needs an OCT report; I'll call SpecialistVLMs.",
        "actions": [
            {
                "API_name": "SpecialistVLMs",
                "API_params": {"image_path": record["image_path"][0]}
            }
        ],
        "value": "Calling SpecialistVLMs to generate the ophthalmic report..."
    }

    final_reply = random.choice(answer_templates).format(report=record["report"])
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"fundus_sample_{idx}",
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
    all_records = load_deepeyenet()
    total = min(len(all_records), n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(tqdm(all_records[:total],
                                       desc=f"Building {total} OCT samples")):
            json.dump(transform(rec, idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Wrote {total} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
