import json
import random
from pathlib import Path
from tqdm import tqdm

IMAGE_DIR = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k")

OUTPUT_FILE = Path("./tool_instruct/healthgpt_reconstruct_dataset.jsonl")

instruction_templates = [
    "Help me reconstruct this image.",
    "I need a reconstructed scan. Can you provide it?",
    "Reconstruct this image, please.",
    "Can you reconstruct an image from scan data?",
    "Reconstruct the imaging result for me.",
    "I'd like to see a reconstructed scan.",
    "I need a sample medical scan reconstructed from this data.",
    "Recreate the image from the case data.",
    "Reconstruct this image that models the raw scan input.",
    "I'd appreciate a visual reconstruction from this scan.",
    "Please produce a scan image using this raw data.",
    "Could you reconstruct the image for review based on this input?",
    "Provide a scan reconstruction for this case.",
    "Could you recreate the image for reconstruction?",
    "I need a reconstructed visualization of this scenario.",
    "Reconstruct a typical scan from this situation.",
    "Please synthesize a diagnostic-style scan from these signals.",
    "I want to see a reconstructed scan example.",
    "Please reconstruct the imaging output.",
    "Produce a scan image based on this case data.",
    "I'd like a simulation of the scan reconstruction.",
    "Create an image based on this clinical case's scan data.",
    "Could you visualize the case in a reconstructed image?",
    "Generate a reconstruction image, please.",
    "Can you generate a reconstructed scan image?",
    "Simulate how the image might appear based on the raw signals.",
    "Please generate a version of the scan from these inputs.",
    "Could you help me visualize the reconstructed imaging outcome?",
    "Show me a reconstructed version of this case.",
    "I'd like to get a reconstructed scan of this situation.",
]

reconstruction_response_templates = [
    "This is the reconstructed image you requested.",
    "Here is the simulated medical image.",
    "The diagnostic image has been generated.",
    "This image illustrates the reconstruction results.",
    "I've completed the image reconstruction.",
    "Here is your synthetic medical scan.",
    "The requested scan has been generated.",
    "Below is the visual reconstruction you asked for.",
    "This image represents the simulated scan.",
    "The reconstructed medical image is now ready.",
    "I've generated the visualization as requested.",
    "Image reconstruction completed as per your request.",
    "Here is the representative medical image.",
    "This scan simulates the requested reconstruction.",
    "Please find the reconstructed image below.",
    "Here's the medical image you asked for.",
    "Synthetic image reconstruction is complete.",
    "The medical scan has been created successfully.",
    "Your reconstructed diagnostic image is ready.",
    "Image generated based on provided medical context."
]

def transform(file_path: Path) -> dict:
    """Generate one conversation record that fits the desired template."""
    # Pick random context
    image_id = file_path.stem
    image_filename = file_path.name
    file_name = str(file_path)

    # 1) Original human prompt
    instruction = random.choice(instruction_templates)
    user_prompt = {
        "from": "human",
        "value": f"<image>\n {instruction}"
    }

    # 2) GPT tool-call
    tool_call = {
        "from": "gpt",
        "thoughts": "To fulfill this request, I'll use the HealthGPT tool for image reconstruction.",
        "actions": [
            {
                "API_name": "HealthGPT",
                "API_params": {
                    "task": "reconstruct_image"
                }
            }
        ],
        "value": "Calling HealthGPT to reconstruct the image..."
    }

    # 3) Penultimate human: return tool output + repeat question
    tool_output = {
        "from": "human",
        "value": (
            f"HealthGPT output: reconstructed image saved as {image_filename}\n\n"
            f"Answer my first request: {instruction}\n\n"
        )
    }

    # 4) Final assistant reply
    final_response = random.choice(reconstruction_response_templates)
    assistant_reply = {
        "from": "gpt",
        "thoughts": "The HealthGPT tool has completed the reconstruction. Now I can answer it based on its output.",
        "actions": [],
        # "value": f"<image>\n{final_response}"
        "value": f"{final_response}"
    }

    return {
        "image_id": image_id,
        "image": image_filename,
        "file_name": file_name,
        "conversations": [
            user_prompt,
            tool_call,
            tool_output,
            assistant_reply
        ]
    }


def build_dataset(n_samples: int = 5000, seed: int = 42, output_path: Path = OUTPUT_FILE) -> None:
    """Generate the dataset by sampling images from IMAGE_DIR and save as JSONL."""
    random.seed(seed)
    # Gather all image files (jpg, png)
    all_images = [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if len(all_images) < n_samples:
        raise ValueError(f"Not enough images in {IMAGE_DIR}: found {len(all_images)}, need {n_samples}")

    sampled = random.sample(all_images, n_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for file_path in tqdm(sampled, desc="Generating reconstruction samples"):
            record = transform(file_path)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Dataset with {n_samples} samples saved to '{output_path}'")

if __name__ == "__main__":
    build_dataset()