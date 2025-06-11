import json
import random
from pathlib import Path
from tqdm import tqdm

IMAGE_DIR = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k")
INPUT_FILE  = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/annotation.json")
OUTPUT_FILE = Path("./tool_instruct/llava_rad_rg_dataset.jsonl")

MODALITIES = ["X-RAY", "CT", "MRI", "US"]

instruction_templates = [
    "Generate a detailed radiology report based on the image.",
    "Review the image and provide a comprehensive diagnostic report.",
    "Interpret this image and summarize all significant findings.",
    "Write a complete radiology report including impressions and recommendations.",
    "Provide an expert diagnostic report based on the following image.",
    "Evaluate the image and draft a structured radiology report.",
    "You are a radiologist. Analyze this image and report your findings.",
    "Compose a diagnostic report covering anatomy, pathology, and relevant observations.",
    "Examine the image and describe any notable abnormalities or variations.",
    "Create a narrative radiology report from the attached scan.",
    "Write a radiologist's report based on the clinical scan provided.",
    "Review the attached scan and produce a professional report.",
    "Create a structured report summarizing the key radiological findings.",
    "Interpret the scan below and generate a report suitable for clinical review.",
    "Write a full report as if preparing for inclusion in a patient chart.",
    "Could you please review this image and generate a report?",
    "Can you interpret the attached image and summarize your findings?",
    "Would you mind providing a radiology report for this image?",
    "Can you help by writing a diagnostic report based on this scan?",
    "Could you describe what this image shows?",
    "Would you be able to draft a full clinical report for this image?",
    "Can you generate a professional radiology summary from this image?",
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
    "Is this image normal or are there signs of pathology?",
    "What can you infer from this scan?",
    "Can you walk me through your radiological interpretation?",
    "What should a radiologist say about this scan?",
    "Do you notice anything concerning in this image?",
    "What might this scan indicate clinically?",
    "You are acting as a clinical radiologist — report on this image.",
    "You're helping with a case review. Please write the radiology findings.",
    "You're a senior radiologist guiding a trainee — show them how you'd report this scan.",
    "Pretend you are documenting for a hospital EMR system — report this scan.",
    "Generate a diagnostic summary to support a referring physician.",
    "You're reviewing this scan for surgical planning — provide your full read.",
    "Simulate an attending radiologist dictating a clinical report.",
    "Write a radiology report as if it will be reviewed in a multidisciplinary team meeting.",
    "You are on a teleradiology shift. Please report the attached image.",
    "This scan was flagged by a junior — provide your expert interpretation.",
    "As a consulting radiologist, provide a second opinion report for this scan.",
    "What does this {modality} image show?",
    "Are there any abnormalities visible in this {modality} scan?",
    "What's your impression of this {modality} image?",
    "Is anything concerning visible in the following {modality} scan?",
    "Can you walk me through your findings from this {modality} image?",
    "Do you notice anything unusual in the attached {modality} scan?",
    "What are the key observations in this {modality} image?",
    "How would you interpret this {modality} scan?",
    "What can be concluded from this {modality} image?",
    "What findings would you report from the provided {modality} scan?",
    "What stands out to you in this {modality} image?",
    "Can you detect any critical findings in this {modality} scan?",
    "Do you see anything that needs follow-up in the {modality} scan?",
    "What is your diagnostic impression from the {modality} image?",
    "Does this {modality} scan suggest any pathology?",
    "Could this {modality} image be interpreted as normal or abnormal?",
    "Could you please generate a radiology report for this {modality} image?",
    "Can you provide a detailed interpretation of the {modality} scan below?",
    "Would you mind writing a radiology report based on this {modality} image?",
    "Could you examine the {modality} image and describe your findings?",
    "Can you summarize the diagnostic findings from the {modality} image?",
    "Could you generate a structured report for the attached {modality} scan?",
    "Would you please interpret this {modality} image and report your impressions?",
    "Can you analyze the following {modality} scan and share a report?",
    "Could you write a complete diagnostic report based on this {modality} image?",
    "Would you be able to produce a radiology report for the attached {modality} scan?",
    "Can you create a radiology summary for this {modality} scan?",
    "May I ask you to provide a clinical report for this {modality} image?",
    "Could you kindly review the {modality} scan and share your thoughts?",
    "Would you assist in interpreting the following {modality} image?",
    "Please could you generate a comprehensive report for the {modality} scan?",
]

answer_templates = [
    "Here is the full {modality} report I generated, including key findings:\n{report}",
    "Below is the detailed radiology report for this {modality} image:\n{report}",
    "I've completed the interpretation. The {modality} report reads as follows:\n{report}",
    "Diagnostic report for the provided {modality} scan:\n{report}",
    "Here's a comprehensive read of the {modality} image:\n{report}",
    "The structured report for this {modality} study is shown below:\n{report}",
    "Find the full {modality} report with impressions and observations:\n{report}",
    "Completed radiology report ({modality}):\n{report}",
    "Full diagnostic findings for the {modality} image:\n{report}",
    "Here is the requested {modality} report, covering all significant details:\n{report}",
    "Below is a narrative report for the {modality} scan:\n{report}",
    "I have drafted the radiology report for your {modality} image:\n{report}",
    "Radiology report (modality: {modality}):\n{report}",
    "Please review the following {modality} report:\n{report}",
    "Here's the in-depth {modality} interpretation:\n{report}",
    "My full written report on the {modality} study is below:\n{report}",
    "The {modality} image has been analyzed. Report:\n{report}",
    "Comprehensive findings for this {modality} scan:\n{report}",
    "Here is an organized {modality} report with findings and impressions:\n{report}",
    "Final {modality} radiology report:\n{report}",
    "Complete diagnostic summary for the {modality} image:\n{report}",
    "I've structured the {modality} report as follows:\n{report}",
    "Kindly review the {modality} report below:\n{report}",
    "Detailed report for your {modality} study:\n{report}",
    "The {modality} findings are summarized here:\n{report}",
    "Attached is the full {modality} radiology report:\n{report}",
    "My interpretation of the {modality} image is as follows:\n{report}",
    "Below is the expert-level {modality} report:\n{report}",
    "Comprehensive {modality} report prepared:\n{report}",
    "Here is the final radiology report for this {modality} scan:\n{report}",
    "Full narrative for the {modality} image:\n{report}",
    "Report generated for {modality} modality:\n{report}",
    "See the detailed {modality} findings below:\n{report}",
    "I have completed the {modality} analysis. Report:\n{report}",
    "Diagnostic impressions for the {modality} study:\n{report}",
    "Radiology report ({modality}) generated:\n{report}",
    "A structured {modality} report has been created:\n{report}",
    "Here is an exhaustive {modality} report:\n{report}",
    "The report for the {modality} image is ready:\n{report}",
    "Please find the {modality} report below:\n{report}",
]

def transform(record: dict, idx: int) -> dict:
    """Convert one raw record to conversation format."""
    image_id_raw = record["image_path"][0]
    image_id = image_id_raw.strip("/").split("/")[-1].split(".")[0]
    image = image_id + ".jpg"
    image_path = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k/" + image_id + ".jpg"
    file_name = str(image_path)

    report_text = record["report"]

    modality = random.choice(MODALITIES)
    instruction = random.choice(instruction_templates).format(modality=modality)
    user_prompt = {
        "from": "human",
        "value": f"<image>\n\n{instruction}"
    }

    tool_call = {
        "from": "gpt",
        "thoughts": "User wants a radiology report; I'll call the LLaVA-Rad tool.",
        "actions": [
            {
                "API_name": "LLaVA-Rad",
                "API_params": {
                    "task": "report_generation",
                    "image_path": image_path
                }
            }
        ],
        "value": "Calling LLaVA-Rad to generate the radiology report..."
    }

    tool_output = {
        "from": "human",
        "value": (
            f"LLaVA-Rad output: {report_text}\n\n"
            f"Answer my first request: {instruction}\n\n"
        )
    }

    friendly_reply = random.choice(answer_templates).format(
        modality=modality, report=report_text
    )
    assistant_reply = {
        "from": "gpt",
        "thoughts": "The LLaVA-Rad tool has completed the report generation task. Now I can answer it based on its output.",
        "actions": [],
        "value": friendly_reply
    }

    return {
        "image_id": image_id,
        "image": image,
        "file_name": file_name,
        "conversations": [
            user_prompt,
            tool_call,
            tool_output,
            assistant_reply
        ]
    }

def build_dataset(input_path: Path = INPUT_FILE,
                  output_path: Path = OUTPUT_FILE,
                  n_samples: int = 5000,
                  seed: int = 42) -> None:
    random.seed(seed)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_records = data.get("train", []) + data.get("validation", []) + data.get("test", [])
    total = min(len(all_records), n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(tqdm(all_records[:total],
                                       desc=f"Building {total} radiology samples")):
            conv = transform(rec, idx)
            json.dump(conv, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Wrote {total} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
