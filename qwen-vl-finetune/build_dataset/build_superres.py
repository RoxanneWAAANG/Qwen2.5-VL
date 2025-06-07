import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/healthgpt_superres_dataset.jsonl")

modalities = ["MRI", "CT", "X-ray", "Ultrasound"]
anatomies = ["brain", "chest", "abdomen", "spine", "liver", "heart", "knee"]
conditions = {
    "brain": ["tumor", "glioma", "stroke", "hydrocephalus", "meningioma"],
    "chest": ["pneumonia", "pulmonary embolism", "lung nodule", "pleural effusion"],
    "abdomen": ["appendicitis", "liver cirrhosis", "pancreatitis", "renal cyst"],
    "spine": ["disc herniation", "spinal stenosis", "scoliosis"],
    "liver": ["cirrhosis", "hepatitis", "hepatocellular carcinoma"],
    "heart": ["myocardial infarction", "left ventricle hypertrophy", "valve regurgitation"],
    "knee": ["ACL tear", "meniscus injury", "ligament tear"]
}

super_res_templates = [
    "Can you make this image clearer?",
    "Please improve the quality of this scan.",
    "The image is blurry—can you enhance it?",
    "I need a sharper version of this scan.",
    "Could you help restore the details in this image?",
    "This scan is hard to interpret. Can you make it clearer?",
    "Is it possible to improve the resolution here?",
    "Can you clean up this fuzzy image?",
    "Can this medical image be made clearer for diagnosis?",
    "Please sharpen the scan for better visibility.",
    "Apply super-resolution to improve the scan clarity.",
    "Run super-resolution on this image to recover fine details.",
    "Upscale the resolution to highlight subtle findings.",
    "Generate a high-resolution version of this diagnostic image.",
    "Perform resolution enhancement to clarify anatomical structures.",
    "Use image enhancement to recover visual fidelity.",
    "Apply a super-resolution pipeline to refine this scan.",
    "Denoise and upscale this image using advanced methods.",
    "Enhance visualization of the {anatomy} showing {condition}.",
    "Recover details in an image of the {anatomy} with suspected {condition}.",
    "Sharpen this scan of the {anatomy} affected by {condition}.",
    "Produce a clearer depiction of {condition} in the {anatomy}.",
    "Improve image quality to better see the {anatomy} and {condition}.",
    "Create a high-detail version of the {anatomy} region with signs of {condition}.",
    "Boost resolution to better visualize the {condition} in the {anatomy}.",
    "Refine image clarity for analyzing {condition} in the {anatomy}.",
    "Make this image of the {anatomy} with {condition} more interpretable.",
    "Enhance fine details in the {anatomy} region to evaluate {condition}.",
    "As a diagnostic assistant, enhance this image for improved clinical review.",
    "Optimize this scan for better interpretation by radiologists.",
    "Produce a super-resolved version for detailed examination.",
    "Help create a high-definition scan suitable for diagnostic purposes.",
    "Render a sharper view of the target region to confirm findings.",
    "Improve this image to aid clinical decision-making.",
    "Make subtle patterns more visible through super-resolution.",
    "Help visualize micro-structures better with enhanced resolution.",
    "Clarify tissue boundaries by improving image fidelity.",
    "Use super-resolution to enhance this image.",
    "Generate a cleaner view of a scan.",
    "Produce a more detailed image.",
    "Refine the visual features for this image case.",
    "Create a clearer scan image.",
]

answer_templates = [
    "Here is your enhanced image:\n{image}",
    "Super-resolution complete. Output image:\n{image}",
    "Image enhancement finished—see result below:\n{image}",
    "The high-resolution scan is ready:\n{image}",
    "Upscaling done. Here is the clarified image:\n{image}",
    "Your requested high-def image:\n{image}",
    "High-quality reconstruction generated:\n{image}",
    "Enhanced diagnostic image below:\n{image}",
    "Detail enhancement complete. Image:\n{image}",
    "The refined scan is provided here:\n{image}",
    "Image sharpening finished. Output:\n{image}",
    "Resolution boost applied. See image:\n{image}",
    "Here's the upgraded scan:\n{image}",
    "HD output created successfully:\n{image}",
    "Enhanced view for clinical review:\n{image}",
    "Your super-resolved image is below:\n{image}",
    "The improved image is now available:\n{image}",
    "Clarity restored—please review:\n{image}",
    "Enhanced {modality} scan attached:\n{image}",
    "Here is the crisp, high-resolution result:\n{image}",
    "Up-scaled image ready:\n{image}",
    "Refined image output:\n{image}",
    "Final high-def image generated:\n{image}",
    "Image quality improved—see below:\n{image}",
    "Enhanced resolution scan:\n{image}",
    "Super-resolution successful. Result:\n{image}",
    "Here's the denoised, sharper image:\n{image}",
    "Completed high-detail reconstruction:\n{image}",
    "The upgraded visual is attached:\n{image}",
    "Sharper diagnostic image:\n{image}",
    "HD reconstruction provided:\n{image}",
    "Improved scan for better evaluation:\n{image}",
    "Pixel enhancement complete. Image:\n{image}",
    "Finalized high-quality output:\n{image}",
    "Here is the high-fidelity scan:\n{image}",
    "Image resolution elevated successfully:\n{image}",
    "Super-resolution pipeline finished:\n{image}",
    "Enhanced spatial detail now available:\n{image}",
    "Upscaled medical image below:\n{image}",
    "Ultra-clear image delivered:\n{image}",
    "High-definition output image:\n{image}",
]

def transform(idx: int) -> dict:
    anatomy   = random.choice(anatomies)
    modality  = random.choice(modalities)
    condition = random.choice(conditions[anatomy])

    user_prompt = random.choice(super_res_templates).format(
        modality=modality, anatomy=anatomy, condition=condition
    )

    tool_call = {
        "from": "gpt",
        "thoughts": (
            f"To enhance the {modality} image of the {anatomy} with {condition}, "
            "I'll call HealthGPT's super_resolution API."
        ),
        "actions": [
            {
                "API_name": "HealthGPT",
                "API_params": {
                    "task": "super_resolution",
                    "modality": modality,
                    "anatomy": anatomy,
                    "condition": condition
                }
            }
        ],
        "value": "Calling HealthGPT super_resolution..."
    }

    final_reply = random.choice(answer_templates).format(
        image="<image>", modality=modality
    )
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"superres_{anatomy}_{idx}",
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples),
                        desc="Generating super-resolution samples"):
            json.dump(transform(idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} super-resolution samples to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
