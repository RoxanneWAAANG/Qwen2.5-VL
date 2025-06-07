import os, json

IMAGE_DIR    = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/5k"
OUT_ANN_PATH = "./tool_instruct/dummy_annotations.json"

# collect all JPEG/PNG files
files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png'))]

# build a list of {“image_path”:…, “caption”:…}
# you can leave "caption" blank or set it to filename:
records = []
for fn in files:
    records.append({
        "image_path": os.path.join(IMAGE_DIR, fn),
        "caption": ""
    })

# write to disk
with open(OUT_ANN_PATH, "w") as f:
    json.dump(records, f, indent=2)
