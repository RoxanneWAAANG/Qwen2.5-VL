import json
import random
from pathlib import Path
from tqdm import tqdm

# Path where your images are stored; update as needed
IMAGE_DIR = "/scratch/project_2002846/Binxu/data_for_classification_2/mri/"

OUTPUT_FILE = Path("./tool_instruct/healthgpt_reconstruct_dataset.jsonl")

modalities = ["MRI", "CT", "X-ray", "Ultrasound"]
anatomies = ["brain", "chest", "abdomen", "spine", "liver", "heart", "knee"]
conditions = {
    "brain": ["glioblastoma", "stroke", "meningioma", "hydrocephalus", "tumor in left frontal lobe"],
    "chest": ["pneumonia", "pulmonary embolism", "lung nodule", "pleural effusion"],
    "abdomen": ["appendicitis", "liver cirrhosis", "pancreatitis", "renal cyst"],
    "spine": ["disc herniation", "spinal stenosis", "scoliosis"],
    "liver": ["hepatocellular carcinoma", "fatty liver"],
    "heart": ["myocardial infarction", "left ventricle hypertrophy"],
    "knee": ["ACL tear", "meniscus injury"]
}

instruction_templates = [
    "Help me reconstruct this image.",
    "I need a reconstructed scan. Can you provide it?",
    "Reconstruct this image, please.",
    # ... (other templates) ...
]

reconstruction_response_templates = [
    "This is the reconstructed image you requested.",
    "Here is the simulated medical image.",
    # ... (other response templates) ...
]

def transform(idx: int) -> dict:
    """Generate one conversation record that fits the desired template."""
    anatomy = random.choice(anatomies)
    condition = random.choice(conditions[anatomy])
    modality = random.choice(modalities)

    image_id = f"reconstruct_{idx}"
    image_filename = f"{image_id}.jpg"
    file_path = f"{IMAGE_DIR}{image_filename}"

    # 1) Original human prompt
    instruction = random.choice(instruction_templates)
    user_prompt = {
        "from": "human",
        "value": f"<image>\n {instruction}"
    }

    # 2) GPT tool‐call
    tool_call = {
        "from": "gpt",
        "thoughts": "To fulfill this request, I'll use the HealthGPT tool for image reconstruction.",
        "actions": [
            {
                "API_name": "HealthGPT",
                "API_params": {
                    "task": "reconstruct_image",
                    "modality": modality,
                    "anatomy": anatomy,
                    "condition": condition
                }
            }
        ],
        "value": "Calling HealthGPT to reconstruct the image..."
    }

    # 3) Penultimate human: returns the tool output + repeats the question
    tool_output = {
        "from": "human",
        "value": (
            f"HealthGPT output: reconstructed image saved as {image_filename}\n\n"
            f"{user_prompt['value']}"
        )
    }

    # 4) Final assistant reply using the new knowledge
    final_response = random.choice(reconstruction_response_templates)
    assistant_reply = {
        "from": "gpt",
        "thoughts": "The HealthGPT tool has completed the reconstruction.",
        "actions": [],
        "value": f"<image>\n{final_response}"
    }

    return {
        "image_id": image_id,
        "image": image_filename,
        "file_name": file_path,
        "conversations": [
            user_prompt,
            tool_call,
            tool_output,
            assistant_reply
        ]
    }

def build_dataset(n_samples: int = 5000, seed: int = 42, output_path: Path = OUTPUT_FILE) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples), desc="Generating reconstruction samples"):
            record = transform(idx)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Dataset with {n_samples} samples saved to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
