#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multi-tool single-round conversations
Format: User request → Assistant uses multiple tools → Tool outputs → Final response

python3 multi_tool_single_round.py \
--tool_yaml corpus_pack/tool_meta.yaml \
--single_round_dir /home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Medical_Agent_Instruction_Tuning/tool_instruct \
--out /home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Medical_Agent_Instruction_Tuning/full_data/multi_tool_single_round.jsonl \
--num 20000
"""

# build_single_round_multitool.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

# ===============================
# Canonical tool names & builtin caps
# ===============================

def canonical_tool_name(name: str) -> str:
    n = (name or "").strip()
    key = n.lower()
    aliases = {
        "conch": "CONCH",
        "dsmil": "DSMIL",
        "cellvit": "CellViT",
        "cellsam": "CellSAM",

        "unigradicon": "UniGradICON",
        "ultrasam": "UltraSAM",
        "healthgpt": "HealthGPT",
        "internet": "IterNet",
        "llava-rad": "LLaVA-Rad",
        "llava_med": "LLaVA-Med",
        "llava-med": "LLaVA-Med",
        "biomedclip": "BiomedClip",
        "medsam": "MedSAM",
        "grounding dino": "grounding dino",
        "grounding dino + medsam": "grounding dino + MedSAM",
        # "chatcad-g": "ChatCAD-G",
        "chatcad-r": "ChatCAD-R",
        "llava": "LLaVA",
        "pmc-llama": "PMC-LLaMA",
        "rate-ner": "RaTE-NER",
        "specialistvlms": "SpecialistVLMs",
    }
    return aliases.get(key, n)

@dataclass
class Tool:
    name: str
    modality: str
    refine_task: str
    default_args: Dict[str, Any]
    refine_responses: List[str]
    success_responses: List[str]
    modalities: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    requires_two_images: bool = False
    image_optional: bool = False

    @classmethod
    def from_dict(cls, name: str, cfg: Dict[str, Any], builtin: Dict[str, Dict[str, Any]]) -> "Tool":
        return cls(
            name=name,
            modality=cfg.get("modality", ""),
            refine_task=cfg.get("refine_task", ""),
            default_args=cfg.get("default_args", {}),
            refine_responses=cfg.get("refine_responses", []),
            success_responses=cfg.get("success_responses", []),
            modalities=cfg.get("modalities") or builtin.get(name, {}).get("modalities", []),
            tasks=cfg.get("tasks") or builtin.get(name, {}).get("tasks", []),
            requires_two_images=cfg.get("requires_two_images", builtin.get(name, {}).get("requires_two_images", False)),
            image_optional=cfg.get("image_optional", builtin.get(name, {}).get("image_optional", False)),
        )

BUILTIN_TOOL_CAPS: Dict[str, Dict[str, Any]] = {
    "UltraSAM": {"modalities": ["US"], "tasks": ["segmentation"]},
    "MedSAM": {"modalities": ["MRI", "CT", "X-ray", "Histology", "Gross"], "tasks": ["segmentation"]},
    "IterNet": {"modalities": ["Retina-Fundus"], "tasks": ["segmentation"]},
    "UniGradICON": {"modalities": ["CT", "MRI"], "tasks": ["registration"], "requires_two_images": True},
    "HealthGPT": {"modalities": ["X-ray", "MRI", "CT", "US"], "tasks": ["reconstruction", "super_resolution"]},
    "LLaVA-Rad": {"modalities": ["X-ray", "MRI", "CT", "US"], "tasks": ["report_generation"]},
    # "ChatCAD-G": {"modalities": ["X-ray"], "tasks": ["report_generation"]},
    "SpecialistVLMs": {"modalities": ["Retina-OCT"], "tasks": ["report_generation"]},
    "BiomedClip": {"modalities": ["MRI", "CT", "X-ray", "Histology", "Gross"], "tasks": ["analysis", "classification"]},
    "LLaVA-Med": {"modalities": ["MRI", "CT", "X-ray", "Histology", "Gross"], "tasks": ["vqa", "analysis"]},
    "grounding dino": {"modalities": ["MRI", "CT", "X-ray", "Histology"], "tasks": ["grounding"]},
    "grounding dino + MedSAM": {"modalities": ["MRI", "CT", "X-ray", "Histology"], "tasks": ["grounded_segmentation", "segmentation"]},
    "LLaVA": {"modalities": [], "tasks": ["summarization"], "image_optional": True},
    "PMC-LLaMA": {"modalities": [], "tasks": ["qa"], "image_optional": True},
    "RaTE-NER": {"modalities": [], "tasks": ["entity_extraction"], "image_optional": True},
    "ChatCAD-R": {"modalities": [], "tasks": ["documentation", "rag"], "image_optional": True},

    # NOTE: Pathology tools (CONCH/DSMIL/CellViT/CellSAM) remain defined for registry compatibility,
    # but are NOT used by the planner anymore.
    "CONCH": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch"], "tasks": ["tissue_classification", "analysis"]},
    "DSMIL": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch"], "tasks": ["tumor_detection", "analysis"]},
    "CellViT": {"modalities": ["Cell-Microscopy"], "tasks": ["cell_segmentation", "analysis"]},
    "CellSAM": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch", "Cell-Microscopy"], "tasks": ["wsi_segmentation", "cell_segmentation", "segmentation"]},
}

class ToolRegistry:
    def __init__(self, yaml_path: Union[str, Path]):
        with Path(yaml_path).open() as f:
            raw = yaml.safe_load(f)
        self.tools: Dict[str, Tool] = {}
        for raw_name, cfg in raw.items():
            canon = canonical_tool_name(raw_name)
            self.tools[canon] = Tool.from_dict(canon, cfg, BUILTIN_TOOL_CAPS)

    def __getitem__(self, name: str) -> Tool:
        return self.tools[canonical_tool_name(name)]

    def has(self, name: str) -> bool:
        return canonical_tool_name(name) in self.tools

    def is_compatible(self, tool_name: str, modality: str) -> bool:
        t = self[tool_name]
        return (not t.modalities) or (modality in t.modalities)

    def tools_for_task_and_modality(self, task: str, modality: str) -> List[str]:
        out = []
        for name, t in self.tools.items():
            if task in t.tasks and (not t.modalities or modality in t.modalities):
                out.append(name)
        return out

# ===============================
# Real data extractor
# ===============================

@dataclass
class ToolExample:
    tool_name: str
    image_id: str
    image_path: Union[str, List[str], None]
    input_prompt: str
    tool_params: Dict[str, Any]
    tool_output: str
    assistant_response: str
    thoughts: str

class RealDataExtractor:
    def __init__(self, tool_instruct_dir: Union[str, Path]):
        self.tool_instruct_dir = Path(tool_instruct_dir)
        self.tool_mapping = {
            "UniGradICON": "unigradicon_reg_dataset.jsonl",
            "UltraSAM": "ultrasam_seg_dataset.jsonl",
            "HealthGPT": "healthgpt_superres_dataset.jsonl",
            "IterNet": "internet_seg_dataset.jsonl",
            "LLaVA-Rad": "llava_rad_rg_dataset.jsonl",
            "LLaVA": "llava_sum_dataset.jsonl",
            "RaTE-NER": "rate_ner_dataset.jsonl",
            "PMC-LLaMA": "pmc_llama_medqa_dataset.jsonl",
            "SpecialistVLMs": "svlms_fundus_dataset.jsonl",
            "CONCH": "CONCH.jsonl",
            "DSMIL-TCGA": "DSMIL-TCGA.jsonl",
            "DSMIL-C16": "DSMIL-C16.jsonl",
            "CellViT": "CellViT.jsonl",
            "CellSAM": "CellSAM.jsonl",
            "LLaVA-Med": "LLaVA-Med.jsonl",
            "BiomedClip": "BiomedClip.jsonl",
            "grounding dino": "GD.jsonl",
            "MedSAM": "MedSAM.jsonl",
            "grounding dino + MedSAM": "GD_MedSAM.jsonl",
            # "ChatCAD-G": "ChatCAD-G.jsonl",
            "ChatCAD-R": "ChatCAD-R.jsonl",
        }
        self._cache: Dict[str, List[ToolExample]] = {}
        self._image_pool: List[str] = []
        self._load_image_pool()

    def _load_image_pool(self) -> None:
        try:
            for tool_name in list(self.tool_mapping.keys()):
                examples = self.load_tool_examples(tool_name, max_examples=10000)
                for ex in examples:
                    if isinstance(ex.image_path, list):
                        for p in ex.image_path:
                            if p and p not in self._image_pool:
                                self._image_pool.append(p)
                    else:
                        if ex.image_path and ex.image_path not in self._image_pool:
                            self._image_pool.append(ex.image_path)
        except Exception as e:
            print(f"Warning: Could not load image pool: {e}")

    def get_different_image(self, exclude_paths: List[str]) -> Optional[str]:
        avail = [img for img in self._image_pool if img not in exclude_paths]
        return random.choice(avail) if avail else None

    def load_tool_examples(self, tool_name: str, max_examples: int = 100000) -> List[ToolExample]:
        tool_name = canonical_tool_name(tool_name)
        if tool_name in self._cache:
            return self._cache[tool_name]
        if tool_name not in self.tool_mapping:
            self._cache[tool_name] = []
            return []
        dataset_file = self.tool_instruct_dir / self.tool_mapping[tool_name]
        if not dataset_file.exists():
            self._cache[tool_name] = []
            return []
        exs: List[ToolExample] = []
        with dataset_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                data = json.loads(line.strip())
                ex = self._parse_example(tool_name, data)
                if ex:
                    exs.append(ex)
        self._cache[tool_name] = exs
        return exs

    def _parse_example(self, expected_tool: str, data: Dict[str, Any]) -> Optional[ToolExample]:
        conversations = data.get("conversations", [])
        if len(conversations) < 3:
            return None
        assistant_call = conversations[1]
        actions = assistant_call.get("actions", [])
        if not actions:
            return None
        actual_tool_name = canonical_tool_name(actions[0].get("API_name", ""))
        if canonical_tool_name(expected_tool) != actual_tool_name:
            return None

        user_prompt = conversations[0]["value"]
        user_prompt = user_prompt.replace("<image>\n", "").replace("<image>", "").strip()

        thoughts = assistant_call.get("thoughts", "")
        tool_params = actions[0].get("API_params", {})
        tool_output_msg = conversations[2]["value"]
        tool_output = tool_output_msg.split("Answer my first request:")[0].strip()
        prefix = f"{actual_tool_name} output:"
        if tool_output.startswith(prefix):
            tool_output = tool_output[len(prefix):].strip()
        assistant_response = conversations[3]["value"] if len(conversations) > 3 else ""

        image_data = data.get("image") or data.get("images")
        image_path: Union[str, List[str], None]
        if isinstance(image_data, list):
            image_path = image_data
        else:
            image_path = image_data

        return ToolExample(
            tool_name=actual_tool_name,
            image_id=data.get("image_id") or " ",
            image_path=image_path,
            input_prompt=user_prompt,
            tool_params=tool_params or {},
            tool_output=tool_output,
            assistant_response=assistant_response,
            thoughts=thoughts,
        )

    def get_random_example(self, tool_name: str) -> Optional[ToolExample]:
        exs = self.load_tool_examples(tool_name)
        return random.choice(exs) if exs else None

# ===============================
# Router (task → tool by modality)
# ===============================

class Router:
    def __init__(self, registry: ToolRegistry):
        self.r = registry

    def pick(self, task: str, modality: str, prefer: Optional[List[str]] = None) -> Optional[str]:
        candidates = self.r.tools_for_task_and_modality(task, modality)
        if prefer:
            pref = [t for t in prefer if t in candidates and self.r.is_compatible(t, modality)]
            if pref:
                return random.choice(pref)
        return random.choice(candidates) if candidates else None

def pick_segmentation(router: Router, modality: str) -> Optional[str]:
    if modality == "US":
        return router.pick("segmentation", modality, prefer=["UltraSAM"])
    if modality == "Retina-Fundus":
        return router.pick("segmentation", modality, prefer=["IterNet"])
    # For CT/MRI/X-ray/Gross
    return router.pick("segmentation", modality, prefer=["MedSAM"])

def pick_report(router: Router, modality: str) -> Optional[str]:
    if modality in ["CT", "MRI", "US", "X-ray"]:
        return router.pick("report_generation", modality, prefer=["LLaVA-Rad"])
    if modality == "Retina-OCT":
        return router.pick("report_generation", modality, prefer=["SpecialistVLMs"])
    # Fundus/Gross: no strict reporter, use summarizer later
    return None

def pick_analysis(router: Router, modality: str) -> Optional[str]:
    # Imaging analysis (no pathology chains)
    if modality in ["CT", "MRI", "X-ray", "Gross"]:
        return router.pick("analysis", modality, prefer=["BiomedClip", "LLaVA-Med"])
    if modality in ["Retina-Fundus"]:
        # could also use BiomedClip/LLaVA-Med depending on your data
        return router.pick("segmentation", modality, prefer=["IterNet"])
    if modality in ["US"]:
        # often segmentation first; fallback to LLaVA-Med
        return router.pick("analysis", modality, prefer=["LLaVA-Med"])
    if modality == "Retina-OCT":
        # OCT specialist reporter exists; analysis step optional
        return router.pick("analysis", modality, prefer=["LLaVA-Med"])
    return router.pick("analysis", modality, prefer=["LLaVA-Med"])

def pick_registration(router: Router, modality: str) -> Optional[str]:
    if modality in ["CT", "MRI"]:
        return router.pick("registration", modality, prefer=["UniGradICON"])
    return None

# ===============================
# Planner for single-round  (PATHOLOGY CHAINS REMOVED)
# ===============================

MODALITY_WEIGHTS = {
    # Removed: Histology / Cell-Microscopy
    "CT": 0.22, "MRI": 0.22, "X-ray": 0.22, "US": 0.14,
    "Retina-Fundus": 0.08, "Retina-OCT": 0.06, "Gross": 0.06,
}

def sample_modality() -> str:
    items, w = zip(*MODALITY_WEIGHTS.items())
    return random.choices(items, weights=w, k=1)[0]

def vague_prompt_for_modality(mod: str, comparative: bool) -> str:
    if comparative:
        return "How did the findings change after therapy?"
    if mod == "US":
        return "What matters most in this ultrasound?"
    if mod in ["CT", "MRI"]:
        return "Please review this scan and explain the key findings."
    if mod == "X-ray":
        return "Please review this chest X-ray and tell me what matters most."
    if mod == "Retina-Fundus":
        return "What are the important findings in this fundus image?"
    if mod == "Retina-OCT":
        return "Please analyze this OCT and summarize what matters."
    if mod == "Gross":
        return "Please analyze this gross specimen image and summarize the key findings."
    return "Please analyze this image and summarize what matters most."

class SingleRoundBuilder:
    def __init__(self, registry: ToolRegistry, extractor: RealDataExtractor,
                 max_steps: int = 3, p_three_steps: float = 0.6, p_comparative_ct_mri: float = 0.25):
        self.r = registry
        self.ex = extractor
        self.router = Router(registry)
        self.max_steps = max_steps
        self.p_three = p_three_steps
        self.p_compare = p_comparative_ct_mri

    def _emit_image_tags(self, n: int) -> str:
        return "".join("<image>\n" for _ in range(max(0, n)))

    def plan_chain(self) -> Dict[str, Any]:
        modality = sample_modality()
        steps: List[str] = []
        comparative = False
        want_three = (random.random() < self.p_three)

        if modality in ["CT", "MRI"] and (random.random() < self.p_compare):
            # Comparative: register -> report -> summarize
            reg = pick_registration(self.router, modality)
            rep = pick_report(self.router, modality)
            steps = [t for t in [reg, rep, "LLaVA"] if t]
            comparative = True

        elif modality == "US":
            # seg -> report -> summarize
            seg = pick_segmentation(self.router, modality)
            rep = pick_report(self.router, modality)
            steps = [t for t in [seg, rep, "LLaVA"] if t]

        elif modality == "X-ray":
            # (seg or analysis) -> report -> summarize
            seg_or_ana = pick_segmentation(self.router, modality) or pick_analysis(self.router, modality)
            rep = pick_report(self.router, modality)
            steps = [t for t in [seg_or_ana, rep, "LLaVA"] if t]

        elif modality in ["CT", "MRI"]:  # non-comparative CT/MRI
            # seg/analysis -> report -> summarize
            seg_or_ana = pick_segmentation(self.router, modality) or pick_analysis(self.router, modality)
            rep = pick_report(self.router, modality)
            steps = [t for t in [seg_or_ana, rep, "LLaVA"] if t]

        elif modality == "Retina-Fundus":
            # IterNet -> LLaVA
            seg = pick_segmentation(self.router, modality)
            steps = [t for t in [seg, "LLaVA"] if t]
            if want_three:
                steps = steps  # keep 2-step; no pathology QA

        elif modality == "Retina-OCT":
            # LLaVA-Med (analysis) -> SpecialistVLMs (report) -> LLaVA
            ana = pick_analysis(self.router, modality)
            spec = pick_report(self.router, modality)  # SpecialistVLMs
            steps = [t for t in [ana, spec, "LLaVA"] if t]
            if not want_three and len(steps) >= 3:
                steps = steps[:2]

        elif modality == "Gross":
            # MedSAM or BiomedClip -> LLaVA
            seg_or_ana = pick_segmentation(self.router, modality) or pick_analysis(self.router, modality)
            steps = [t for t in [seg_or_ana, "LLaVA"] if t]

        # Trim to 2 or 3
        if len(steps) >= 3 and not want_three:
            steps = steps[:2]
        if len(steps) < 2:
            # ensure at least two tools
            if modality in ["CT", "MRI"]:
                steps = [t for t in [pick_analysis(self.router, modality), pick_report(self.router, modality)] if t]
                if len(steps) < 2:
                    steps = [t for t in [pick_analysis(self.router, modality), "LLaVA"] if t]
            else:
                steps = [t for t in [pick_analysis(self.router, modality), "LLaVA"] if t]

        return {"modality": modality, "tools": steps, "comparative": comparative}

    def build_entry(self) -> Dict[str, Any]:
        plan = self.plan_chain()
        tools = plan["tools"]
        modality = plan["modality"]
        comparative = plan["comparative"]
        if len(tools) < 2:
            return {}

        # Sample first tool example to anchor image/id
        first_ex = self.ex.get_random_example(tools[0])
        if not first_ex:
            return {}

        # Decide images (2 if UniGradICON)
        first_img = self._first_image(first_ex.image_path)
        image_list: List[str] = [first_img] if first_img else []
        if "UniGradICON" in tools:
            second = self.ex.get_different_image(image_list)
            if not second:
                return {}
            image_list.append(second)

        # User prompt with image tags
        prompt_core = vague_prompt_for_modality(modality, comparative)
        if len(image_list) == 2:
            user_value = f"{self._emit_image_tags(1)}{self._emit_image_tags(1)}{prompt_core}"
        else:
            user_value = f"{self._emit_image_tags(1)}{prompt_core}"
        user_turn = {"from": "human", "value": user_value}

        # Assistant plan (names tools explicitly in value)
        actions, names = [], []
        for i, tool in enumerate(tools):
            # Prefer sampling params from the tool's own example
            ex_for_params = self.ex.get_random_example(tool) or first_ex
            step = {"API_name": tool, "API_params": ex_for_params.tool_params}
            if i > 0:
                step["depends_on"] = [tools[i-1]]
            actions.append(step)
            names.append(tool)

        plan_thoughts, plan_value = self._plan_text(names, modality, comparative)
        asst_turn2 = {
            "from": "gpt",
            "thoughts": plan_thoughts,
            "actions": actions,
            "value": plan_value
        }

        # Human tool outputs (concatenate in the same order)
        outputs_blocks = []
        sampled_examples: List[ToolExample] = []
        for tool in tools:
            ex = self.ex.get_random_example(tool)
            if not ex:
                return {}
            sampled_examples.append(ex)
            out = ex.tool_output.strip() or ex.assistant_response.strip() or "(no output)"
            if len(out) > 1200:
                out = out[:1200] + " ... <truncated>"
            outputs_blocks.append(f"{tool} output: {out}")
        human_outputs_value = "\n\n".join(outputs_blocks) + f"\n\nAnswer my first request: {prompt_core}"
        human_turn3 = {"from": "human", "value": human_outputs_value}

        # Final assistant answer
        last_ex = sampled_examples[-1]
        final_value = last_ex.assistant_response.strip() or self._synthesize_final_value(tools, modality)
        final_thoughts = f"Based on the outputs of {', '.join(tools)}, I can provide a comprehensive answer."
        asst_turn4 = {"from": "gpt", "thoughts": final_thoughts, "actions": [], "value": final_value}

        conversations = [user_turn, asst_turn2, human_turn3, asst_turn4]

        if not self._validate(conversations, image_list, tools, modality):
            return {}

        top_image_field: Union[str, List[str]] = image_list[0] if len(image_list) == 1 else image_list
        return {
            # "session_id": str(uuid.uuid4()),
            # "image_id": first_ex.image_id,
            "image": top_image_field,
            # "file_name": image_list[0] if image_list else None,
            "conversations": conversations
        }

    # ---------- helpers ----------
    def _first_image(self, image_path: Union[str, List[str], None]) -> Optional[str]:
        if isinstance(image_path, list):
            return image_path[0] if image_path else None
        return image_path

    def _plan_text(self, tool_names: List[str], modality: str, comparative: bool) -> (str, str):
        if comparative:
            thoughts = "Register the two scans, generate a comparison report, then summarize the overall change."
        else:
            thoughts = "Run a logical multi-tool chain and then synthesize a concise answer."
        if len(tool_names) == 2:
            value = f"I'll use {tool_names[0]} first, then {tool_names[1]} to complete the task."
        else:
            value = f"I'll use {tool_names[0]}, then {tool_names[1]}, and finally {tool_names[2]} to complete the task."
        return thoughts, value

    def _synthesize_final_value(self, tools: List[str], modality: str) -> str:
        if "UniGradICON" in tools:
            return "Registration shows interval change; the comparison report indicates response without new concerning findings."
        if modality == "US":
            return "Key structures are segmented, the report is consistent with the imaging appearance, and the summary reflects the main abnormality."
        if modality == "X-ray":
            return "No acute cardiopulmonary process is identified; correlate with symptoms and follow-up as needed."
        if modality in ["CT", "MRI"]:
            return "Findings are summarized based on analysis and structured reporting; consider clinical correlation and follow-up."
        if modality == "Retina-Fundus":
            return "Segmentation-driven assessment highlights key retinal findings; suggest ophthalmology follow-up."
        if modality == "Retina-OCT":
            return "OCT analysis and the specialist-style report indicate the primary features; follow clinical guidance."
        if modality == "Gross":
            return "Salient macroscopic features are identified and concisely summarized for clinical correlation."
        return "Findings are summarized based on tool outputs and suggest appropriate next steps."

    def _validate(self, conversations: List[Dict[str, Any]], image_list: List[str], tools: List[str], modality: str) -> bool:
        # Tags equal images
        user_msg = conversations[0]["value"]
        if user_msg.count("<image>") != len(image_list):
            return False
        # UniGradICON rules
        if "UniGradICON" in tools:
            if modality not in ["CT", "MRI"] or len(image_list) != 2:
                return False
        # Ensure 4 turns + actions + Answer line
        if len(conversations) != 4:
            return False
        if not conversations[1].get("actions"):
            return False
        if "Answer my first request:" not in conversations[2].get("value", ""):
            return False
        # No pathology chains exist here; nothing else to block
        return True

# ===============================
# CLI
# ===============================

def main():
    ap = argparse.ArgumentParser(description="Build single-round multi-tool dataset (no pathology chains).")
    ap.add_argument("--tool_yaml", type=str, required=True, help="Path to tool metadata YAML")
    ap.add_argument("--single_round_dir", type=str, required=True, help="Path to single-round examples directory")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL file")
    ap.add_argument("--num", type=int, required=True, help="Number of samples to generate")
    ap.add_argument("--max_steps", type=int, default=3, help="Max tools per round (2 or 3)")
    ap.add_argument("--p_three_steps", type=float, default=0.4, help="Probability of 3-tool chains")
    ap.add_argument("--p_compare", type=float, default=0.25, help="Probability of comparative chain for CT/MRI")
    args = ap.parse_args()

    registry = ToolRegistry(args.tool_yaml)
    extractor = RealDataExtractor(args.single_round_dir)
    builder = SingleRoundBuilder(registry, extractor,
                                 max_steps=args.max_steps,
                                 p_three_steps=args.p_three_steps,
                                 p_comparative_ct_mri=args.p_compare)

    print("Loading tool examples...")
    total = 0
    for tool_name in registry.tools:
        n = len(extractor.load_tool_examples(tool_name, max_examples=40))
        print(f"{tool_name}: loaded {n} examples")
        total += n
    print(f"Total examples loaded: {total}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num):
            entry = builder.build_entry()
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                success += 1
            else:
                skipped += 1
            if (i + 1) % 100 == 0:
                print(f"Progress {i+1}/{args.num} | Success {success} | Skipped {skipped}")

    print("\nFinal Results:")
    print(f"Generated: {success} valid")
    print(f"Success rate: {success / max(args.num,1) * 100:.1f}%")
    print(f"Total skipped: {skipped}")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
