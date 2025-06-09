import json, random
from tqdm import tqdm
from pathlib import Path
import os

# pip install pycocotools
from pycocotools.coco import COCO

DATA_ROOT = '/home/jack/Projects/yixin-llm/yixin-llm-data/UltraSam/dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023'
IMG_PATH = '/home/jack/Projects/yixin-llm/yixin-llm-data/UltraSam/dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/images/'
ANN_FILE = os.path.join(DATA_ROOT, 'annotations/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023__coco.json')
OUTPUT_FILE = Path("./tool_instruct/ultrasam_seg_dataset.jsonl")
NUM_SAMPLES = 5000

templates = [
    "Segment the region within the box {bbox} in this ultrasound image:",
    "Please segment the area inside bounding box {bbox} of the ultrasound frame:",
    "Isolate the portion defined by {bbox} in this ultrasound image:",
    "Delineate the structure within bounding box {bbox} on ultrasound image.",
    "Run segmentation on the crop {bbox} of the ultrasound scan:.",
    "Extract the region {bbox} from ultrasound image and produce a mask.",
    "Perform pixel-wise segmentation inside box {bbox} in.",
    "Segment the rectangle {bbox} in the ultrasound frame.",
    "Segment only the area bounded by {bbox} in this ultrasound image:.",
    "Generate a mask for the box {bbox} region in.",
    "Outline the area inside {bbox} on ultrasound image.",
    "Segment the rectangle defined by {bbox} in this ultrasound scan:.",
    "Segment the ROI at {bbox} in the ultrasound image.",
    "Segment the area within coordinates {bbox} in.",
    "Isolate and mask the region {bbox} in ultrasound image.",
    "Crop and segment the box {bbox} from the ultrasound frame.",
    "Identify and segment the bounding box {bbox} on.",
    "Outline the lesion inside {bbox} in this ultrasound scan:.",
    "Perform segmentation for the sub-image defined by {bbox} in.",
    "Extract a binary mask for box {bbox} in ultrasound image.",
    "Segment the pathology within {bbox} on.",
    "Delineate tissue boundaries inside {bbox} of this ultrasound image:.",
    "Please segment the specified box {bbox} in the ultrasound frame.",
    "Perform boundary detection within {bbox} on ultrasound image.",
    "Segment the structure at {bbox} in this ultrasound image:.",
    "Mask the region {bbox} in ultrasound scan.",
    "Segment the pixels inside box {bbox} on.",
    "Segmentation request: focus on the area {bbox} in.",
    "Precisely segment the area bounded by {bbox} in.",
    "Generate a mask for the ROI at {bbox} on ultrasound image.",
    "Partition the pixels within {bbox} in this ultrasound scan:.",
    "Segment the subregion {bbox} of ultrasound image.",
    "Generate a segmentation mask for coordinates {bbox} on.",
    "Detect and segment the zone {bbox} in this ultrasound frame:.",
    "Focus segmentation on the box {bbox} in ultrasound image.",
    "Perform ultrasound segmentation limited to {bbox} in.",
    "Extract and segment the region {bbox} in.",
    "Segment, isolating the area indicated by bboxes={bbox}.",
    "Outline the contours within {bbox} on ultrasound image.",
    "Segment only the region inside {bbox} in this scan.",
    "Segment the target anatomy inside box {bbox} for ultrasound image.",
    "Highlight and mask the region defined by {bbox} in.",
    "Extract the ROI at {bbox} from ultrasound frame.",
    "Apply segmentation to the clipped area {bbox} of image.",
    "Isolate and segment {bbox} in this ultrasound scan:.",
    "Perform focused segmentation on the area {bbox} in.",
    "Run segmentation on ultrasound image, concentrating on bounding box {bbox}.",
    "Segment the region inside {bbox} in.",
    "Generate an accurate mask for coordinates {bbox} on ultrasound image."
]

answer_templates = [
    "Segmentation completed for the specified box. Here is the mask:\n{mask}",
    "UltraSAM has generated the segmentation mask:\n{mask}",
    "Below is the binary mask for the region of interest:\n{mask}",
    "Here is the segmented output focusing on the given bbox:\n{mask}",
    "The ROI within the box is now segmented. Mask:\n{mask}",
    "Mask for the requested bounding box:\n{mask}",
    "The target region has been isolated; see mask below:\n{mask}",
    "Below find the segmentation result for the bbox:\n{mask}",
    "Segmentation mask produced by UltraSAM:\n{mask}",
    "The region inside the bounding box is segmented. Mask:\n{mask}",
    "Here's the output segmentation mask:\n{mask}",
    "Completed segmentation mask for the ultrasound ROI:\n{mask}",
    "The ROI mask has been generated:\n{mask}",
    "Binary segmentation of specified region:\n{mask}",
    "Please review the generated mask for the bbox:\n{mask}",
    "UltraSAM mask for the selected area:\n{mask}",
    "Final segmentation mask attached:\n{mask}",
    "Segmentation task finished. Output mask:\n{mask}",
    "Here is the delineated lesion mask:\n{mask}",
    "Mask showing segmented region:\n{mask}",
    "The ROI within the ultrasound image is now masked:\n{mask}",
    "Segmented output for the specified rectangle:\n{mask}",
    "Mask for the bbox area generated:\n{mask}",
    "Segmentation complete—mask provided below:\n{mask}",
    "Output image with segmented ROI:\n{mask}",
    "Here is the binary mask isolating the lesion:\n{mask}",
    "Below is the bbox-focused segmentation mask:\n{mask}",
    "UltraSAM successfully produced the mask:\n{mask}",
    "Segmentation overlay ready:\n{mask}",
    "Final mask for the designated region:\n{mask}",
    "The requested area has been segmented:\n{mask}",
    "Here's the refined mask for the bbox:\n{mask}",
    "Segmentation result obtained. Mask:\n{mask}",
    "Bounded region mask generated:\n{mask}",
    "Here is the lesion segmentation mask:\n{mask}",
    "Mask file for the ROI is provided:\n{mask}",
    "The ultrasound bounding box is segmented. Mask:\n{mask}",
    "Segmentation mask output below:\n{mask}",
    "Region inside the bbox isolated:\n{mask}",
    "Here's the produced segmentation mask:\n{mask}",
    "Completed ROI segmentation:\n{mask}",
]

coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

def transform(idx: int) -> dict:
    image_id = random.choice(img_ids)
    ann_id = random.choice(coco.getAnnIds(imgIds=image_id))
    ann    = coco.loadAnns([ann_id])[0]
    img    = coco.loadImgs([image_id])[0]
    image = img["file_name"]
    file_name = IMG_PATH + img["file_name"]

    x, y, w, h = ann["bbox"]
    bbox = [x, y, x + w, y + h]

    instruction = random.choice(templates).format(bbox=bbox)
    user_prompt = {
        "from": "human",
        "value": f"<image>\n\n{instruction}"
    }

    tool_call = {
        "from": "gpt",
        "thoughts": "This is an ultrasound segmentation task; I'll call the UltraSAM tool with a bbox prompt.",
        "actions": [
            {
                "API_name": "UltraSAM",
                "API_params": {
                    "image": image_id,
                    "prompt": {"bboxes": [bbox]}
                }
            }
        ],
        "value": "Calling UltraSAM to segment the ultrasound image within the specified box..."
    }

    tool_output = {
        "from": "human",
        "value": (
            f"UltraSAM output: {image}\n\n"
            f"Answer my first request: {instruction}\n\n"
        )
    }

    final_reply = random.choice(answer_templates).format(mask=file_name)
    assistant_reply = {
        "from": "gpt",
        "thoughts": "The UltraSAM tool has completed the image segmentation task. Now I can answer it based on its output.",
        "actions": [],
        "value": final_reply
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

def build_dataset(n_samples: int = 5000,
                  seed: int = 42,
                  output_path: Path = OUTPUT_FILE) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples),
                        desc="Generating UltraSAM bbox samples"):
            json.dump(transform(idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} UltraSAM bbox records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
