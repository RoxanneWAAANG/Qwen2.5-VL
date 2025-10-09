#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a JSON file into multiple JSONL files by API_name,
and drop unnecessary fields ("image_id", "file_name").

Usage:
    python split_by_api.py input.json

It will create a folder `tools_split/` and save files like:
    tools_split/CellSAM.jsonl
    tools_split/OtherTool.jsonl
"""

import os
import sys
import json

def split_json_by_api(input_file, output_dir="/home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Medical_Agent_Instruction_Tuning/tool_instruct"):
    os.makedirs(output_dir, exist_ok=True)
    api_files = {}

    # load full JSON (list of objects)
    with open(input_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for data in data_list:
        # keep only the fields we want
        filtered = {
            "image": data.get("image"),
            "conversations": data.get("conversations", [])
        }

        # extract API_name from first available action
        api_name = None
        for conv in filtered["conversations"]:
            if "actions" in conv and conv["actions"]:
                api_name = conv["actions"][0].get("API_name")
                break

        if api_name is None:
            continue  # skip if no tool found

        # open file handle if not yet created
        if api_name not in api_files:
            file_path = os.path.join(output_dir, f"{api_name}.jsonl")
            api_files[api_name] = open(file_path, "w", encoding="utf-8")

        # write each entry as JSONL line
        api_files[api_name].write(json.dumps(filtered, ensure_ascii=False) + "\n")

    # close all files
    for f in api_files.values():
        f.close()

    print(f"Done! Files saved in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_by_api.py input.json")
        sys.exit(1)

    input_file = sys.argv[1]
    split_json_by_api(input_file)
