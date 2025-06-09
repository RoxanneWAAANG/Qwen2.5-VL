import json
import random
from pathlib import Path
from tqdm import tqdm

IMAGE_DIR = Path("/home/jack/.cache/kagglehub/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset/versions/10/full-fundus/full-fundus")
OUTPUT_FILE = Path("./tool_instruct/internet_seg_dataset.jsonl")

vessel_segmentation_templates = [
    "Segment the retinal blood vessels in this fundus image.",
    "Please identify the vessel network in this retina image.",
    "Can you generate a segmentation mask of blood vessels?",
    "Extract the retinal vasculature from this scan.",
    "I need the blood vessels segmented in this retinal image.",
    "Can you analyze the vascular structure in this fundus photo?",
    "Generate the segmentation overlay for this retina image.",
    "Detect and segment the vasculature in this fundus image.",
    "Create a vessel map from the following image.",
    "I'd like to see the segmented vessels in this scan.",
    "Please process this image for vessel segmentation.",
    "Identify blood vessels in the retina photo.",
    "Extract the vasculature mask from this input image.",
    "Segment all visible vessels in this retinal scan.",
    "Can you show the segmented vessels from this fundus image?",
    "Generate a vessel probability map for this image.",
    "Create a binary map of vessels from this scan.",
    "Visualize the vessel tree from this retinal image.",
    "I want to segment the retinal vessels in this image.",
    "Please isolate the vessels in the retina scan.",
    "Detect and extract vessels from this fundus photo.",
    "Can you create a vessel mask from this image?",
    "Segment the vasculature in this input image.",
    "Get the blood vessel segmentation for this scan.",
    "What do the segmented vessels look like in this image?",
    "Extract vascular regions from the following retina image.",
    "I'd like a map of the vessels in this image.",
    "Highlight the vasculature in this retinal scan.",
    "Please return the vessel segmentation mask.",
    "Segment the main arteries and veins in this image.",
    "Trace the blood vessels in this scan.",
    "Show the segmented microvasculature from this image.",
    "Create an overlay of the vessel segmentation.",
    "Analyze and mark vessel paths in the image.",
    "Return a segmentation of the vascular network.",
    "Extract the visible vessel tree from this image.",
    "Please perform segmentation of vasculature.",
    "I'd like to visualize the blood vessels from this scan.",
    "Mark all vascular branches in the retina image.",
    "Provide the segmented output of retinal vessels.",
    "What are the vessels in this fundus scan?",
    "Create a vascular mask overlay on this image.",
    "Can you extract the vascular tree?",
    "I need to segment the vessel paths in this scan.",
    "Please analyze this retinal image for vessels.",
    "Give me the vessel map for this input image.",
    "Provide a vessel segmentation mask from this retina image.",
    "Identify and highlight all vessels in this image.",
    "Generate a vessel segmentation result.",
    "Can you help segment the blood vessels in this image?",
    "Segment the vessels in this fundus image to demonstrate how retinal vasculature appears.",
    "Help annotate this fundus photo for vessel analysis.",
    "Prepare the vessel segmentation as part of the retinal screening workflow.",
    "Segment the vasculature in this image to assist in diagnosing diabetic retinopathy.",
    "Identify retinal vessels to check for ischemia in this image.",
    "Please outline the vascular network for potential grading.",
    "Extract all major and minor vessels for triage review.",
    "Trace all visible vessels and return a clear segmentation.",
    "Create a map of vessels that can be used for branching structure analysis.",
    "Focus on segmenting fine microvasculature in this image.",
    "Extract only the visible capillary network.",
    "Highlight small branching vessels in this retinal image.",
    "I'd like to visualize where the blood vessels are in my eye using this image.",
    "Help me understand the blood vessel layout in this retinal photo.",
    "Use this photo of the retina to show the blood vessels clearly.",
]

vessel_segmentation_responses = [
    "The vessel segmentation is complete. Here is the mask: {mask}",
    "IterNet has generated the retinal vessel mask: {mask}",
    "Below is the binary vessel segmentation result: {mask}",
    "Here is the segmented vasculature overlay: {mask}",
    "Completed vessel segmentation; mask attached: {mask}",
    "Vessel mask produced by IterNet: {mask}",
    "Retinal vasculature successfully segmented: {mask}",
    "Here's the output vessel mask: {mask}",
    "Segmentation finished—see mask below: {mask}",
    "Resulting vessel segmentation image: {mask}",
    "The extracted vascular network is provided here: {mask}",
    "Binary vessel map generated: {mask}",
    "IterNet output mask: {mask}",
    "Vessel segmentation completed: {mask}",
    "Please review the vessel mask: {mask}",
    "Here is the final vessel segmentation: {mask}",
    "Segmentation mask for retinal vessels: {mask}",
    "The vascular tree has been isolated: {mask}",
    "Here's the delineated vessel network: {mask}",
    "Mask showing segmented vessels: {mask}",
    "Retinal vessel segmentation image: {mask}",
    "The retinal vasculature mask is attached: {mask}",
    "Below find the segmented vessel overlay: {mask}",
    "IterNet segmentation result: {mask}",
    "Output image with vessel mask: {mask}",
    "Final vessel segmentation mask: {mask}",
    "Here is the isolated vascular structure: {mask}",
    "Binary mask for retinal vessels: {mask}",
    "Completed vessel map: {mask}",
    "Here is the vascular segmentation output: {mask}",
    "Segmentation mask provided: {mask}",
    "The vessels have been segmented; see below: {mask}",
    "Here's the extracted vasculature mask: {mask}",
    "Vessel segmentation (IterNet): {mask}",
    "Result mask of retinal vessels: {mask}",
    "Below is the IterNet vessel segmentation: {mask}",
    "Segmentation overlay ready: {mask}",
    "Here is the detailed vessel mask: {mask}",
    "The retinal vessel map is as follows: {mask}",
    "Segmentation success—mask below: {mask}",
    "Here is the generated vessel segmentation mask: {mask}",
]

def transform(file_path: Path) -> dict:
    """
    Generate one conversation record for retinal vessel segmentation.
    
    This function creates a complete four-turn conversation that demonstrates
    how to request vessel segmentation, call the IterNet tool, and interpret
    the results. Each conversation uses a real fundus image from the dataset.
    
    Args:
        file_path: Path to the fundus image file that will be processed
        
    Returns:
        Dictionary containing the complete conversation record with image metadata
    """
    # Extract image metadata from the file path
    # This ensures each conversation corresponds to a real image file
    image_id = file_path.stem  # Filename without extension becomes the ID
    image_filename = file_path.name  # Full filename with extension
    file_name = str(file_path)  # Complete file path as string

    # 1) Initial human request with image reference
    # Select a random vessel segmentation request from our templates
    instruction = random.choice(vessel_segmentation_templates)
    user_prompt = {
        "from": "human",
        "value": f"<image>\n{instruction}"
    }

    # 2) Assistant response with IterNet tool call
    # This demonstrates the proper way to call IterNet for vessel segmentation
    tool_call = {
        "from": "gpt",
        "thoughts": "This is a retinal vessel segmentation task. I'll use the IterNet tool to analyze the fundus image and generate a vessel segmentation mask.",
        "actions": [
            {
                "API_name": "IterNet",
                "API_params": {
                    "task": "segment_retinal_vessels",
                    "image_path": file_name,
                    "output_format": "binary_mask"
                }
            }
        ],
        "value": "Calling IterNet to segment retinal vessels in the fundus image..."
    }

    # 3) Human message with tool output and request reiteration
    # This simulates the tool returning results and asking for interpretation
    tool_output = {
        "from": "human",
        "value": (
            f"IterNet output: vessel segmentation mask saved as {image_filename}\n\n"
            f"Answer my first request: {instruction}\n\n"
        )
    }

    # 4) Final assistant response with formatted result
    # Select a random response template and format it with the output image reference
    final_response = random.choice(vessel_segmentation_responses).format(mask="<output_image>")
    assistant_reply = {
        "from": "gpt",
        "thoughts": "The IterNet tool has completed the retinal vessel segmentation task and generated a binary mask. Now I can provide the formatted result to the user.",
        "actions": [],
        "value": final_response
    }

    # Return the complete conversation record with all required metadata
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
    """
    Build the retinal vessel segmentation dataset by sampling fundus images.
    
    This function creates a complete dataset of vessel segmentation conversations
    using real fundus images from the specified directory. Each conversation
    demonstrates the complete workflow from initial request through tool usage
    to final result interpretation.
    
    Args:
        n_samples: Number of conversation examples to generate
        seed: Random seed for reproducible sampling
        output_path: Where to save the generated dataset
    """
    # Set random seed for reproducible dataset generation
    random.seed(seed)
    
    # Find all valid image files in the dataset directory
    # We look for common medical image formats used in fundus photography
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    all_images = [
        p for p in IMAGE_DIR.iterdir() 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    
    # Verify we have enough images for the requested dataset size
    if len(all_images) < n_samples:
        raise ValueError(
            f"Not enough images in {IMAGE_DIR}: found {len(all_images)}, need {n_samples}"
        )

    print(f"Found {len(all_images)} fundus images in {IMAGE_DIR}")
    
    # Randomly sample the requested number of images for our dataset
    # This ensures we get a diverse subset if the directory contains more images than needed
    sampled_images = random.sample(all_images, n_samples)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate conversation records and save them in JSONL format
    # Each line in the output file will contain one complete conversation
    with output_path.open("w", encoding="utf-8") as fout:
        for file_path in tqdm(sampled_images, desc="Generating vessel segmentation conversations"):
            # Create a conversation record for this fundus image
            record = transform(file_path)
            
            # Write the record as a JSON line (JSONL format)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"\nDataset generation complete!")
    print(f"Created {n_samples} vessel segmentation conversations")
    print(f"Dataset saved to: {output_path}")
    print(f"Each conversation demonstrates IterNet tool usage for retinal vessel analysis")

if __name__ == "__main__":
    # Build the dataset with default parameters
    # You can modify these parameters based on your specific requirements
    build_dataset(
        n_samples=5000,  # Number of training examples to generate
        seed=42,         # Random seed for reproducible results
        output_path=OUTPUT_FILE  # Where to save the generated dataset
    )
