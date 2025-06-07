import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/unigradicon_reg_dataset.jsonl")
NUM_SAMPLES = 5000
MODALITIES  = ["CT", "MRI"]

PROMPT_TEMPLATES = [
    "Can you align the moving image at '{moving}' to the fixed image at '{fixed}' for slice {slice_idx} in {modality}?",
    "Help me register this {modality} scan: moving='{moving}', fixed='{fixed}', slice={slice_idx}.",
    "Please perform registration between these scans. Fixed: '{fixed}', Moving: '{moving}', Slice: {slice_idx}, Modality: {modality}.",
    "I need the transform to align moving to fixed image (slice {slice_idx}, {modality}).",
    "Could you register the moving file to the fixed one at slice {slice_idx} using {modality} images?",
    "Register: fixed='{fixed}', moving='{moving}', {modality}, slice={slice_idx}.",
    "Compute transform for {modality}, slice {slice_idx}: moving → fixed.",
    "Align {modality} images at slice {slice_idx}.",
    "Apply slice-wise registration (slice {slice_idx}, modality={modality}).",
    "Run registration for '{moving}' to '{fixed}' — slice {slice_idx}, {modality}.",
    "Align these {modality} scans to correct for patient motion. Slice: {slice_idx}.",
    "Use image registration to enable follow-up analysis on slice {slice_idx}.",
    "Apply registration for better alignment before segmentation. Inputs: '{moving}', '{fixed}', slice {slice_idx}, modality={modality}.",
    "Perform registration to fuse time-point scans. Focus: slice {slice_idx}, modality={modality}.",
    "Compute deformation field to align moving '{moving}' to fixed '{fixed}' on {modality} slice {slice_idx}.",
    "You're given two images: moving and fixed. Align them (slice {slice_idx}, {modality}).",
    "Using the registration model, match the moving scan '{moving}' to the reference '{fixed}' on slice {slice_idx}.",
    "Execute registration between fixed and moving volumes on slice {slice_idx}.",
    "Calculate the registration result to align these volumes: '{moving}' → '{fixed}', slice={slice_idx}, modality={modality}.",
    "Estimate alignment transform for the provided image pair — modality: {modality}, slice: {slice_idx}.",
    "Compute the spatial transform from moving ('{moving}') to fixed ('{fixed}') on slice {slice_idx} of {modality} volume.",
    "Generate registration parameters to align slice {slice_idx} of {modality} data from moving to fixed image.",
    "Derive the transformation matrix that maps '{moving}' to '{fixed}' at slice {slice_idx}, modality={modality}.",
    "Estimate deformable registration field between moving and fixed volumes on slice {slice_idx}.",
    "Align these {modality} scans and return the slice-level registration output (slice {slice_idx}).",
    "Align moving to fixed. Slice: {slice_idx}, modality: {modality}.",
    "Register scan '{moving}' to '{fixed}', slice {slice_idx}.",
    "Apply image registration on slice {slice_idx}.",
    "Run volume registration (modality={modality}, slice={slice_idx}).",
    "Calculate registration: {modality}, slice {slice_idx}.",
    "Use '{fixed}' as reference and align '{moving}' to it (modality: {modality}, slice: {slice_idx}).",
    "Register moving file to fixed image using {modality} slice {slice_idx}.",
    "Match '{moving}' to '{fixed}' on slice {slice_idx} for {modality} modality.",
    "Align files: moving='{moving}', fixed='{fixed}', modality={modality}, slice={slice_idx}.",
    "Perform rigid/deformable registration between '{moving}' and '{fixed}', slice {slice_idx}.",
    "Try to align the moving image to the fixed one using registration (slice {slice_idx}, modality={modality}).",
    "We need to register these scans: '{moving}' to '{fixed}' at slice {slice_idx}, {modality}.",
    "Align this slice for image fusion: moving='{moving}', fixed='{fixed}', slice {slice_idx}.",
    "Run registration to assist with image interpretation. Modality: {modality}, slice: {slice_idx}.",
    "Use registration to prepare the scans for downstream analysis. Inputs: '{moving}', '{fixed}', slice {slice_idx}.",
]

answer_templates = [
    "The registration for slice {slice_idx} on {modality} data is complete. Key alignment metrics look good.",
    "UniGradICON has produced the warp field for slice {slice_idx} ({modality}). Review the output image for accuracy.",
    "Registration finished: slice {slice_idx}, modality {modality}. The transformed moving image should now overlay well.",
    "Here is the registered result for slice {slice_idx} in {modality}. Misalignment has been minimized.",
    "Alignment completed on {modality} slice {slice_idx}. Inspect anatomical landmarks to confirm quality.",
    "UniGradICON successfully aligned the moving scan to the fixed scan for slice {slice_idx} ({modality}).",
    "The transformation parameters for slice {slice_idx} ({modality}) have been calculated and applied.",
    "Slice {slice_idx} ({modality}) has been registered. Examine the fused overlay for verification.",
    "Registration is successful for {modality} slice {slice_idx}. The output image reflects the new coordinate mapping.",
    "Warp computation done on {modality} slice {slice_idx}. The moving volume now matches the fixed reference.",
    "The output image shows slice {slice_idx} ({modality}) after UniGradICON registration—alignment looks consistent.",
    "Transform matrix for {modality} slice {slice_idx} is applied. The images should now coincide anatomically.",
    "Finished aligning slice {slice_idx} in {modality}. Please review regions of interest for residual offsets.",
    "Registration task complete: {modality}, slice {slice_idx}. Output appears well-aligned.",
    "Slice {slice_idx} ({modality}) registered with minimal distortion. Check overlay for subtle shifts.",
    "UniGradICON produced a deformation field for {modality} slice {slice_idx}. The result is attached.",
    "Alignment achieved for slice {slice_idx} ({modality}). Verify critical structures in the output.",
    "Here's the registered {modality} slice {slice_idx}. Visual inspection suggests good correspondence.",
    "The moving image is now aligned to the fixed image for {modality} slice {slice_idx}.",
    "Registration parameters applied to slice {slice_idx} ({modality}). Output ready for evaluation.",
    "Completed registration on slice {slice_idx} in {modality}. Overlay appears accurate.",
    "Warp field successfully generated for {modality} slice {slice_idx}. Alignment metrics are within tolerance.",
    "Slice {slice_idx} ({modality}) alignment finished. Significant structures line up correctly.",
    "The registration workflow finalized for {modality} slice {slice_idx}. Review the result image.",
    "Transform estimation complete for slice {slice_idx}, modality {modality}.",
    "UniGradICON alignment executed on slice {slice_idx} ({modality}). Examine fused view for confirmation.",
    "All set—slice {slice_idx} of the {modality} dataset has been registered.",
    "Registration done for {modality} slice {slice_idx}. Key features align as expected.",
    "The UniGradICON model has aligned slice {slice_idx} ({modality}). Quality seems satisfactory.",
    "Output for slice {slice_idx} ({modality}) is generated after registration.",
    "Slice {slice_idx} in {modality} modality registered. You may proceed with further analysis.",
    "Registration complete—{modality} slice {slice_idx}. Overlay inspection recommended.",
    "Deformation field applied to slice {slice_idx} ({modality}). Alignment confirmed.",
    "The moving scan now matches the fixed scan at slice {slice_idx} ({modality}).",
    "The registration of {modality} slice {slice_idx} has converged successfully.",
    "Alignment parameters for slice {slice_idx} ({modality}) computed and saved.",
    "Slice {slice_idx} ({modality}) looks well-aligned post-registration. Verify if needed.",
    "UniGradICON finished processing slice {slice_idx} ({modality}). Output provided.",
    "Here is the registered image for slice {slice_idx} ({modality}).",
    "Registration complete for slice {slice_idx} ({modality}). Review and proceed.",
    "Slice {slice_idx} ({modality}) registration succeeded with acceptable residual error."
]

def transform(idx: int) -> dict:
    modality   = random.choice(MODALITIES)
    slice_idx  = random.randint(0, 99)
    fixed_path = "<fixed_image>"
    moving_path = "<moving_image>"

    user_prompt = random.choice(PROMPT_TEMPLATES).format(
        fixed=fixed_path, moving=moving_path,
        modality=modality, slice_idx=slice_idx
    )

    tool_call = {
        "from": "gpt",
        "thoughts": "This is an image-registration task; I'll call the UniGradICON tool.",
        "actions": [
            {
                "API_name": "UniGradICON",
                "API_params": {
                    "fixed_path": fixed_path,
                    "moving_path": moving_path,
                    "modality": modality,
                    "slice_idx": slice_idx
                }
            }
        ],
        "value": "Calling UniGradICON to register images..."
    }

    final_answer = random.choice(answer_templates).format(
        modality=modality, slice_idx=slice_idx
    )
    assistant_reply = {
        "from": "gpt",
        "value": f"<output_image>\n{final_answer}"
    }

    return {
        "id": f"unigradicon_reg_{idx}",
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
                        desc="Generating UniGradICON instructions"):
            record = transform(idx)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
