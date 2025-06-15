import json
import random
from pathlib import Path
from tqdm import tqdm


OUTPUT_FILE = Path("./tool_instruct/unigradicon_reg_dataset.jsonl")
NUM_SAMPLES = 5000
MODALITIES  = ["CT", "MRI"]

PROMPT_TEMPLATES = [
    "Can you align the moving image to the fixed image for {modality}?",
    "Help me register this {modality} scan.",
    "Please perform registration between these scans in {modality}.",
    "I need the transform to align moving to fixed image in {modality}.",
    "Could you register the moving file to the fixed one using {modality} images?",
    "Register these {modality} images.",
    "Compute transform for {modality}: moving → fixed.",
    "Align {modality} images.",
    "Apply image registration for {modality}.",
    "Run registration for {modality}.",
    "Align these {modality} scans to correct for patient motion.",
    "Use image registration to enable follow-up analysis on {modality}.",
    "Apply registration for better alignment before segmentation in {modality}.",
    "Perform registration to fuse time-point scans for {modality}.",
    "Compute deformation field to align moving to fixed on {modality}.",
    "You're given two images: moving and fixed. Align them for {modality}.",
    "Using the registration model, match the moving scan to the reference in {modality}.",
    "Execute registration between fixed and moving volumes in {modality}.",
    "Calculate the registration result to align these volumes in {modality}.",
    "Estimate alignment transform for the provided image pair in {modality}.",
    "Compute the spatial transform from moving to fixed for {modality} volume.",
    "Generate registration parameters to align {modality} data from moving to fixed image.",
    "Derive the transformation matrix for {modality}.",
    "Estimate deformable registration field between moving and fixed volumes in {modality}.",
    "Align these {modality} scans and return the registration output.",
    "Align moving to fixed for {modality}.",
    "Register these scans for {modality}.",
    "Apply image registration for {modality}.",
    "Run volume registration for {modality}.",
    "Calculate registration for {modality}.",
    "Use fixed as reference and align moving to it for {modality}.",
    "Register moving file to fixed image using {modality}.",
    "Match moving to fixed for {modality} modality.",
    "Align files for {modality}.",
    "Perform rigid/deformable registration for {modality}.",
    "Try to align the moving image to the fixed one using registration for {modality}.",
    "We need to register these scans in {modality}.",
    "Align this for image fusion in {modality}.",
    "Run registration to assist with image interpretation for {modality}.",
    "Use registration to prepare the scans for downstream analysis in {modality}.",
    
    # Generic prompts without modality specification
    "Can you align the moving image to the fixed image?",
    "Help me register these medical scans.",
    "Please perform registration between these two images.",
    "I need the transform to align moving to fixed image.",
    "Could you register the moving file to the fixed one?",
    "Register these medical images.",
    "Compute transform: moving → fixed.",
    "Align these medical images.",
    "Apply image registration.",
    "Run registration on these images.",
    "Align these scans to correct for patient motion.",
    "Use image registration to enable follow-up analysis.",
    "Apply registration for better alignment before segmentation.",
    "Perform registration to fuse time-point scans.",
    "Compute deformation field to align moving to fixed.",
    "You're given two images: moving and fixed. Please align them.",
    "Using the registration model, match the moving scan to the reference.",
    "Execute registration between fixed and moving volumes.",
    "Calculate the registration result to align these volumes.",
    "Estimate alignment transform for the provided image pair.",
    "Compute the spatial transform from moving to fixed.",
    "Generate registration parameters to align data from moving to fixed image.",
    "Derive the transformation matrix.",
    "Estimate deformable registration field between moving and fixed volumes.",
    "Align these scans and return the registration output.",
    "Align moving to fixed.",
    "Register these medical scans.",
    "Apply image registration.",
    "Run volume registration.",
    "Calculate registration.",
    "Use fixed as reference and align moving to it.",
    "Register moving file to fixed image.",
    "Match moving to fixed.",
    "Align these files.",
    "Perform rigid/deformable registration.",
    "Try to align the moving image to the fixed one using registration.",
    "We need to register these scans.",
    "Align this for image fusion.",
    "Run registration to assist with image interpretation.",
    "Use registration to prepare the scans for downstream analysis.",
    "Please help me align these two medical images.",
    "I have two images that need to be registered.",
    "Can you perform image alignment on these scans?",
    "These images need spatial alignment.",
    "Help me correct the misalignment between these images.",
    "I need these two images to be spatially matched.",
    "Can you warp the moving image to match the fixed one?",
    "Please compute the transformation between these images.",
    "These medical images need to be co-registered.",
    "Can you align these images for better comparison?",
    "I need to register these two scans.",
    "Please perform spatial alignment on these images.",
    "Can you match these two medical images?",
    "Help me align these images for analysis.",
    "These scans need registration for proper overlay.",
    "Can you correct the spatial differences between these images?",
    "I need these images aligned for accurate measurement.",
    "Please register these medical images.",
    "Can you align these scans for fusion?",
    "Help me match the geometry of these two images.",
]

tool_output_templates = [
    "UniGradICON output: Registration completed successfully for {modality}. Transformation matrix calculated and applied.",
    "UniGradICON output: Image alignment finished for {modality}. Deformation field generated.",
    "UniGradICON output: Registration process complete. {modality} images aligned with minimal residual error.",
    "UniGradICON output: Successful registration for {modality}. Warp field computed and applied.",
    "UniGradICON output: Alignment completed for {modality}. Moving image transformed to match fixed image.",
    "UniGradICON output: Registration finished. {modality} processed successfully.",
    "UniGradICON output: Image registration complete for {modality}. Transformation parameters optimized.",
    "UniGradICON output: Alignment successful. {modality} images now spatially registered.",
    "UniGradICON output: Registration workflow completed for {modality}. Output images generated.",
    "UniGradICON output: Deformation field calculated for {modality}. Registration successful.",
    "UniGradICON output: Registration completed successfully. Transformation matrix calculated and applied.",
    "UniGradICON output: Image alignment finished. Deformation field generated.",
    "UniGradICON output: Registration process complete. Images aligned with minimal residual error.",
    "UniGradICON output: Successful registration. Warp field computed and applied.",
    "UniGradICON output: Alignment completed. Moving image transformed to match fixed image.",
    "UniGradICON output: Registration finished. Images processed successfully.",
    "UniGradICON output: Image registration complete. Transformation parameters optimized.",
    "UniGradICON output: Alignment successful. Images now spatially registered.",
    "UniGradICON output: Registration workflow completed. Output images generated.",
    "UniGradICON output: Deformation field calculated. Registration successful.",
    "UniGradICON output: Medical image registration completed with high accuracy.",
    "UniGradICON output: Spatial alignment achieved. Images are now co-registered.",
    "UniGradICON output: Registration task finished. Moving image successfully warped to fixed image.",
    "UniGradICON output: Image registration process completed. Transformation applied successfully.",
    "UniGradICON output: Alignment procedure finished. Images are now spatially matched.",
]

answer_templates = [
    "Based on the UniGradICON registration results, the {modality} images have been successfully aligned. The transformation parameters have been calculated and applied.",
    "The registration is complete for {modality}. UniGradICON has successfully aligned the moving image to the fixed image with good spatial correspondence.",
    "According to the UniGradICON output, the alignment for {modality} is successful. The deformation field has been computed and the images are now registered.",
    "The registration process for {modality} has been completed successfully. The UniGradICON model has generated the appropriate transformation to align the images.",
    "Based on the results, {modality} has been registered effectively. The moving image has been transformed to match the spatial coordinates of the fixed image.",
    "The UniGradICON registration for {modality} is complete. The alignment shows good correspondence between anatomical structures.",
    "Registration successful for {modality}. The transformation matrix has been applied and the images are now spatially aligned.",
    "The alignment process for {modality} has finished. UniGradICON has successfully computed the deformation field for registration.",
    "Based on the UniGradICON output, the registration for {modality} is complete with satisfactory alignment metrics.",
    "The registration workflow for {modality} has been successfully executed. The images are now properly aligned for further analysis.",
    "Based on the UniGradICON registration results, the images have been successfully aligned. The transformation parameters have been calculated and applied.",
    "The registration is complete. UniGradICON has successfully aligned the moving image to the fixed image with good spatial correspondence.",
    "According to the UniGradICON output, the alignment is successful. The deformation field has been computed and the images are now registered.",
    "The registration process has been completed successfully. The UniGradICON model has generated the appropriate transformation to align the images.",
    "Based on the results, the images have been registered effectively. The moving image has been transformed to match the spatial coordinates of the fixed image.",
    "The UniGradICON registration is complete. The alignment shows good correspondence between anatomical structures.",
    "Registration successful. The transformation matrix has been applied and the images are now spatially aligned.",
    "The alignment process has finished. UniGradICON has successfully computed the deformation field for registration.",
    "Based on the UniGradICON output, the registration is complete with satisfactory alignment metrics.",
    "The registration workflow has been successfully executed. The images are now properly aligned for further analysis.",
    "The medical images have been successfully co-registered. Spatial alignment is now achieved between the moving and fixed images.",
    "Registration completed successfully. The images are now spatially matched and ready for comparative analysis.",
    "The image alignment task has been completed. The moving image has been warped to match the geometry of the fixed image.",
    "Successful registration achieved. The transformation has been applied and the images are now properly aligned.",
    "The registration process finished successfully. Both images are now in the same coordinate system for accurate comparison.",
]

def generate_image_paths(modality: str) -> list:
    """Generate placeholder image paths for fixed and moving images"""
    return [
        f"medical_images/{modality.lower()}/fixed_image.jpg",
        f"medical_images/{modality.lower()}/moving_image.jpg"
    ]

def transform(idx: int) -> dict:
    modality = random.choice(MODALITIES)
    
    # Generate image paths
    image_paths = generate_image_paths(modality)
    
    # Choose a prompt template
    chosen_template = random.choice(PROMPT_TEMPLATES)
    
    # Generate user prompt - handle both templates with and without modality
    try:
        # Try to format with modality
        formatted_prompt = chosen_template.format(modality=modality)
    except KeyError:
        # If no {modality} placeholder, use template as is
        formatted_prompt = chosen_template
    
    user_prompt = f"{formatted_prompt}"
    
    # Generate tool call response
    tool_call_response = {
        "from": "gpt",
        "thoughts": "This is an image registration task. I need to use UniGradICON to align the moving image to the fixed image.",
        "actions": [
            {
                "API_name": "UniGradICON",
                "API_params": {
                    "modality": modality
                }
            }
        ],
        "value": "I'll use the UniGradICON model to perform image registration and align these medical images."
    }
    
    # Generate tool output
    chosen_tool_template = random.choice(tool_output_templates)
    try:
        # Try to format with modality
        tool_output = chosen_tool_template.format(modality=modality)
    except KeyError:
        # If no {modality} placeholder, use template as is
        tool_output = chosen_tool_template
    
    human_tool_response = f"{tool_output}\n\nAnswer my original question about the image registration."
    
    # Generate final answer
    chosen_answer_template = random.choice(answer_templates)
    try:
        # Try to format with modality
        final_answer = chosen_answer_template.format(modality=modality)
    except KeyError:
        # If no {modality} placeholder, use template as is
        final_answer = chosen_answer_template
    
    return {
        "images": image_paths,
        "conversations": [
            {
                "from": "human",
                "value": '<image>\n<image>\n' + user_prompt
            },
            tool_call_response,
            {
                "from": "human",
                "value": (
                    f"UniGradICON output: {image_paths[0]}\n\n"
                    f"Answer my first request: {user_prompt}\n\n"
                )
            },
            {
                "from": "gpt",
                "thoughts": "Based on the UniGradICON output, I can now provide a comprehensive answer about the registration results.",
                "actions": [],
                "value": final_answer
            }
        ]
    }

def build_dataset(n_samples: int = 5000,
                  seed: int = 42,
                  output_path: Path = OUTPUT_FILE) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples),
                        desc="Generating UniGradICON instructions"):
            record = transform(idx)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
