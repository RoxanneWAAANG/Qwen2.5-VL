import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/internet_seg_dataset.jsonl")

templates_internet = [
    "Segment the retinal blood vessels in this fundus image: <image>",
    "Please identify the vessel network in the retina image: <image>",
    "Can you generate a segmentation mask of blood vessels in <image>?",
    "Extract the retinal vasculature from this scan: <image>",
    "I need the blood vessels segmented in this retinal image: <image>",
    "Can you analyze the vascular structure in this fundus photo: <image>?",
    "Generate the segmentation overlay for this retina image: <image>",
    "Detect and segment the vasculature in this fundus image: <image>",
    "Create a vessel map from the following image: <image>",
    "I'd like to see the segmented vessels in <image>",
    "Please process this image for vessel segmentation: <image>",
    "Identify blood vessels in the retina photo: <image>",
    "Extract the vasculature mask from the input image: <image>",
    "Segment all visible vessels in this retinal scan: <image>",
    "Can you show the segmented vessels from this fundus image: <image>?",
    "Generate a vessel probability map for image: <image>",
    "Create a binary map of vessels from <image>",
    "Visualize the vessel tree from this scan: <image>",
    "I want to segment the retinal vessels in this image: <image>",
    "Please isolate the vessels in the retina scan: <image>",
    "Detect and extract vessels from this fundus photo: <image>",
    "Can you create a vessel mask from the image: <image>?",
    "Segment the vasculature in this input image: <image>",
    "Get the blood vessel segmentation for this scan: <image>",
    "What do the segmented vessels look like in <image>?",
    "Extract vascular regions from the following retina image: <image>",
    "I'd like a map of the vessels in this image: <image>",
    "Highlight the vasculature in this retinal scan: <image>",
    "Please return the vessel segmentation mask for <image>",
    "Segment the main arteries and veins in this image: <image>",
    "Trace the blood vessels in this scan: <image>",
    "Show the segmented microvasculature from this image: <image>",
    "Create an overlay of the vessel segmentation for <image>",
    "Analyze and mark vessel paths in the image: <image>",
    "Return a segmentation of the vascular network in <image>",
    "Extract the visible vessel tree from image: <image>",
    "Please perform segmentation of vasculature in <image>",
    "I'd like to visualize the blood vessels from this scan: <image>",
    "Mark all vascular branches in the retina image: <image>",
    "Provide the segmented output of retinal vessels: <image>",
    "What are the vessels in this fundus scan? <image>",
    "Create a vascular mask overlay on this image: <image>",
    "Can you extract the vascular tree from <image>?",
    "I need to segment the vessel paths in this scan: <image>",
    "Please analyze this retinal image for vessels: <image>",
    "Give me the vessel map for this input image: <image>",
    "Provide a vessel segmentation mask from this retina image: <image>",
    "Identify and highlight all vessels in this image: <image>",
    "Generate a vessel segmentation result for: <image>",
    "Can you help segment the blood vessels in this image: <image>?",
    "What do the retinal vessels look like in this image? Please segment them.",
    "Show me where the vessels are in this retinal image: <image>",
    "Could you identify the vessel tree in this scan: <image>?",
    "Segment vessels: <image>",
    "Fundus vessel mask: <image>",
    "Get vasculature from: <image>",
    "Retinal vessel extraction -> <image>",
    "Segment the vessels in this fundus image to demonstrate how retinal vasculature appears.",
    "Use this image to show vessel segmentation <image>",
    "Help annotate this fundus photo for vessel analysis: <image>",
    "Prepare the vessel segmentation as part of the retinal screening workflow: <image>",
    "Segment the vasculature in this image to assist in diagnosing DR or RVO: <image>",
    "Identify retinal vessels to check for ischemia in this image: <image>",
    "Please outline the vascular network for potential grading: <image>",
    "Extract all major and minor vessels for triage review: <image>",
    "Trace all visible vessels and return a clear segmentation: <image>",
    "Perform vessel segmentation: <image>",
    "Create a map of vessels that can be used for branching structure analysis: <image>",
    "Focus on segmenting fine microvasculature in this image: <image>",
    "Extract only the visible capillary network in <image>",
    "Highlight small branching vessels in this retinal image: <image>",
    "I'd like to visualize where the blood vessels are in my eye — use <image>",
    "Help me understand the blood vessel layout in this image: <image>",
    "Use this photo of the retina to show the blood vessels clearly: <image>",
]

answer_templates = [
    "The vessel segmentation is complete. Here is the mask:\n{mask}",
    "IterNet has generated the retinal vessel mask:\n{mask}",
    "Below is the binary vessel segmentation result:\n{mask}",
    "Here is the segmented vasculature overlay:\n{mask}",
    "Completed vessel segmentation; mask attached:\n{mask}",
    "Vessel mask produced by IterNet:\n{mask}",
    "Retinal vasculature successfully segmented:\n{mask}",
    "Here's the output vessel mask:\n{mask}",
    "Segmentation finished—see mask below:\n{mask}",
    "Resulting vessel segmentation image:\n{mask}",
    "The extracted vascular network is provided here:\n{mask}",
    "Binary vessel map generated:\n{mask}",
    "IterNet output mask:\n{mask}",
    "Vessel segmentation completed:\n{mask}",
    "Please review the vessel mask:\n{mask}",
    "Here is the final vessel segmentation:\n{mask}",
    "Segmentation mask for retinal vessels:\n{mask}",
    "The vascular tree has been isolated:\n{mask}",
    "Here's the delineated vessel network:\n{mask}",
    "Mask showing segmented vessels:\n{mask}",
    "Retinal vessel segmentation image:\n{mask}",
    "The retinal vasculature mask is attached:\n{mask}",
    "Below find the segmented vessel overlay:\n{mask}",
    "IterNet segmentation result:\n{mask}",
    "Output image with vessel mask:\n{mask}",
    "Final vessel segmentation mask:\n{mask}",
    "Here is the isolated vascular structure:\n{mask}",
    "Binary mask for retinal vessels:\n{mask}",
    "Completed vessel map:\n{mask}",
    "Here is the vascular segmentation output:\n{mask}",
    "Segmentation mask provided:\n{mask}",
    "The vessels have been segmented; see below:\n{mask}",
    "Here's the extracted vasculature mask:\n{mask}",
    "Vessel segmentation (IterNet):\n{mask}",
    "Result mask of retinal vessels:\n{mask}",
    "Below is the IterNet vessel segmentation:\n{mask}",
    "Segmentation overlay ready:\n{mask}",
    "Here is the detailed vessel mask:\n{mask}",
    "The retinal vessel map is as follows:\n{mask}",
    "Segmentation success—mask below:\n{mask}",
    "Here is the generated vessel segmentation mask:\n{mask}",
]

def transform(idx: int) -> dict:
    user_prompt = random.choice(templates_internet)

    tool_call = {
        "from": "gpt",
        "thoughts": "This is a retinal vessel segmentation task; I'll call the IterNet tool.",
        "actions": [
            {"API_name": "IterNet", "API_params": {"image": "<image>"}}
        ],
        "value": "Calling IterNet to segment retinal vessels..."
    }

    final_reply = random.choice(answer_templates).format(mask="<output_image>")
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"internet_seg_{idx}",
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
                        desc="Generating IterNet vessel segmentation samples"):
            json.dump(transform(idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} IterNet records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
