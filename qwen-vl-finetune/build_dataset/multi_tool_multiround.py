# -*- coding: utf-8 -*-
'''
python3 multi_tool_multiround.py \
  --tool_yaml corpus_pack/tool_meta.yaml \
  --single_round_dir /home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Medical_Agent_Instruction_Tuning/tool_instruct \
  --out /home/jack/Projects/yixin-llm/yixin-llm-data/multi_round/Medical_Agent_Instruction_Tuning/full_data/multi_tool_multiround.jsonl \
  --num 100000
'''
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import yaml
import re

# ===============================
# Scenario & Chain Definitions
# ===============================

class ConversationScenario(Enum):
    SINGLE_IMAGE = "single_image"                    # Scenario 1
    REGISTRATION_FROM_START = "registration_from_start"  # Scenario 2
    REGISTRATION_ADD_LATER = "registration_add_later"    # Scenario 3
    SWITCH_IMAGE_MID = "switch_image_mid"                # Scenario 4

class ChainType(Enum):
    COMPLETE_IMAGING = "complete_imaging"   # Chain 1（已去掉 QA）
    DIAGNOSTIC = "diagnostic"              # Chain 2
    COMPARATIVE = "comparative"            # Chain 3
    # 短链（含 QA）
    REPORT_SUMMARY_QA = "report_summary_qa"    # 报告 → 总结 → QA
    ANALYSIS_QA = "analysis_qa"                # 分析 → QA
    SEGMENT_SUMMARY_QA = "segment_summary_qa"  # 分割 → 总结 → QA
    # 新增：随机混合（示例：先 QA，再注册；中途换图、任务不相关）
    RANDOM_MIX = "random_mix"                  # QA → Registration（两图）

# Chain templates expressed as TASKS (tools are chosen via routing per modality)
CHAIN_TASKS: Dict[ChainType, List[str]] = {
    # 去掉 QA：现在 4 步
    ChainType.COMPLETE_IMAGING: [
        "registration", "segmentation", "report_generation", "summarization"
    ],
    ChainType.DIAGNOSTIC: [
        "image_enhancement", "analysis", "specialist_review", "entity_extraction", "documentation"
    ],
    ChainType.COMPARATIVE: [
        "image_a_analysis", "image_b_analysis", "registration", "comparison_report", "clinical_summary"
    ],
    # 短链（含 QA）
    ChainType.REPORT_SUMMARY_QA: [
        "report_generation", "summarization", "qa"
    ],
    ChainType.ANALYSIS_QA: [
        "analysis", "qa"
    ],
    ChainType.SEGMENT_SUMMARY_QA: [
        "segmentation", "summarization", "qa"
    ],
    # 随机混合：先 QA，再 Registration（满足你“换了个图做不相关注册”的例子）
    ChainType.RANDOM_MIX: [
        "qa", "registration"
    ],
}

# ===============================
# Canonical tool names & mapping
# ===============================
def canonical_tool_name(name: str) -> str:
    n = (name or "").strip()
    key = n.lower()
    aliases = {
        "conch": "CONCH",
        "dsmil": "DSMIL",
        "dsmil-tcga": "DSMIL-TCGA",
        "dsmil_tcga": "DSMIL-TCGA",
        "dsmil-c16": "DSMIL-C16",
        "dsmil_c16": "DSMIL-C16",
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

# ===============================
# Tool Metadata
# ===============================

@dataclass
class Tool:
    name: str
    modality: str
    refine_task: str
    default_args: Dict[str, Any]
    refine_responses: List[str]
    success_responses: List[str]

    # New fields (backward-compatible; inferred when absent)
    modalities: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    requires_two_images: bool = False
    image_optional: bool = False

    @classmethod
    def from_dict(cls, name: str, cfg: Dict[str, Any], builtin: Dict[str, Dict[str, Any]]) -> "Tool":
        modalities = cfg.get("modalities") or builtin.get(name, {}).get("modalities", [])
        tasks = cfg.get("tasks") or builtin.get(name, {}).get("tasks", [])
        requires_two_images = cfg.get("requires_two_images", builtin.get(name, {}).get("requires_two_images", False))
        image_optional = cfg.get("image_optional", builtin.get(name, {}).get("image_optional", False))
        return cls(
            name=name,
            modality=cfg.get("modality", ""),
            refine_task=cfg.get("refine_task", ""),
            default_args=cfg.get("default_args", {}),
            refine_responses=cfg.get("refine_responses", []),
            success_responses=cfg.get("success_responses", []),
            modalities=modalities,
            tasks=tasks,
            requires_two_images=requires_two_images,
            image_optional=image_optional,
        )

# Built-in routing knowledge (used if YAML lacks modalities/tasks)
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
    # Pathology family
    "CONCH": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch"], "tasks": ["tissue_classification", "analysis"]},
    "DSMIL": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch"], "tasks": ["tumor_detection", "analysis"]},
    "DSMIL-TCGA": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch"], "tasks": ["tumor_detection", "analysis"]},
    "DSMIL-C16":  {"modalities": ["Histology", "Histology-WSI", "Histology-Patch"], "tasks": ["tumor_detection", "analysis"]},
    "CellViT": {"modalities": ["Cell-Microscopy"], "tasks": ["cell_segmentation", "analysis"]},
    "CellSAM": {"modalities": ["Histology", "Histology-WSI", "Histology-Patch", "Cell-Microscopy"], "tasks": ["wsi_segmentation", "cell_segmentation", "segmentation"]},
}

# ===============================
# Tool Registry
# ===============================

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
# Real Data Extraction
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
        # Canonical tool name -> dataset filename
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
            "DSMIL": "DSMIL.jsonl",
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
                examples = self.load_tool_examples(tool_name, max_examples=100000)
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
        if tool_output.startswith(f"{actual_tool_name} output:"):
            tool_output = tool_output[len(f"{actual_tool_name} output:"):].strip()
        assistant_response = conversations[3]["value"] if len(conversations) > 3 else ""

        image_data = data.get("image") or data.get("images")
        if isinstance(image_data, list):
            image_path = image_data
        else:
            image_path = image_data

        return ToolExample(
            tool_name=actual_tool_name,
            image_id=data.get("image_id") or " ",
            image_path=image_path,
            input_prompt=user_prompt,
            tool_params=tool_params,
            tool_output=tool_output,
            assistant_response=assistant_response,
            thoughts=thoughts,
        )

    def get_random_example(self, tool_name: str) -> Optional[ToolExample]:
        exs = self.load_tool_examples(tool_name)
        return random.choice(exs) if exs else None

# ===============================
# Conversation State & Artifacts
# ===============================

@dataclass
class Artifact:
    id: str
    type: str
    source_tool: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConvState:
    session_id: str
    scenario: ConversationScenario
    chain_type: ChainType
    modality: str
    base_image_id: Optional[str] = None
    base_image_path: Optional[str] = None
    all_image_paths: List[str] = field(default_factory=list)
    has_second_image: bool = False
    second_image_added_at_turn: int = -1
    image_switch_turn: int = -1
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    conversation_context: List[str] = field(default_factory=list)
    tool_history: List[str] = field(default_factory=list)

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.id] = artifact

    def get_artifacts_by_type(self, artifact_type: str) -> List[Artifact]:
        return [a for a in self.artifacts.values() if a.type == artifact_type]

    def add_context(self, context: str) -> None:
        if context:
            self.conversation_context.append(context)

# ===============================
# Router (task → tool by modality)
# ===============================

class Router:
    """Encapsulates modality-aware routing for tasks."""
    def __init__(self, registry: ToolRegistry):
        self.r = registry

    def pick(self, task: str, modality: str, prefer: Optional[List[str]] = None) -> Optional[str]:
        candidates = self.r.tools_for_task_and_modality(task, modality)
        if prefer:
            pref = [t for t in prefer if t in candidates and self.r.is_compatible(t, modality)]
            if pref:
                return random.choice(pref)
        return random.choice(candidates) if candidates else None

    def route_chain_task(self, chain_task: str, modality: str) -> Optional[str]:
        # Chain 1 (Complete Imaging)
        if chain_task == "registration":
            return self.pick("registration", modality, prefer=["UniGradICON"])
        if chain_task == "segmentation":
            if modality == "US":
                return self.pick("segmentation", modality, prefer=["UltraSAM"])
            if modality == "Retina-Fundus":
                return self.pick("segmentation", modality, prefer=["IterNet"])
            if modality in ["Histology", "Histology-WSI", "Cell-Microscopy", "Histology-Patch"]:
                return self.pick("segmentation", modality, prefer=["CellSAM", "MedSAM"])
            return self.pick("segmentation", modality, prefer=["MedSAM"])
        if chain_task == "report_generation":
            if modality in ["CT", "MRI", "US", "X-ray"]:
                return self.pick("report_generation", modality, prefer=["LLaVA-Rad"])
            if modality == "Retina-OCT":
                return self.pick("report_generation", modality, prefer=["SpecialistVLMs"])
            return None
        if chain_task == "summarization":
            return self.pick("summarization", modality, prefer=["LLaVA"])
        if chain_task == "qa":
            return self.pick("qa", modality, prefer=["PMC-LLaMA"])

        # Chain 2 (Diagnostic Workflow)
        if chain_task == "image_enhancement":
            if modality in ["X-ray", "CT", "MRI", "US"]:
                return self.pick("reconstruction", modality, prefer=["HealthGPT"])
            return None
        if chain_task == "analysis":
            if modality in ["Histology", "Histology-WSI", "Histology-Patch"]:
                return self.pick("analysis", modality, prefer=["CONCH", "DSMIL", "DSMIL-TCGA", "DSMIL-C16", "CellSAM"])
            if modality == "Cell-Microscopy":
                return self.pick("analysis", modality, prefer=["CellViT", "CellSAM"])
            return self.pick("analysis", modality, prefer=["BiomedClip", "LLaVA-Med"])
        if chain_task == "specialist_review":
            if modality == "Retina-OCT":
                return self.pick("report_generation", modality, prefer=["SpecialistVLMs"])
            if modality in ["X-ray", "CT", "MRI", "US"]:
                return self.pick("report_generation", modality, prefer=["LLaVA-Rad"])
            return self.pick("qa", modality, prefer=["PMC-LLaMA"])
        if chain_task == "entity_extraction":
            return self.pick("entity_extraction", modality, prefer=["RaTE-NER"])
        if chain_task == "documentation":
            return self.pick("documentation", modality, prefer=["ChatCAD-R"])

        # Chain 3 (Comparative)
        if chain_task in ["image_a_analysis", "image_b_analysis"]:
            if modality in ["CT", "MRI", "X-ray"]:
                return self.pick("analysis", modality, prefer=["BiomedClip", "MedSAM"])
            return self.pick("analysis", modality, prefer=["BiomedClip"])
        if chain_task == "comparison_report":
            if modality in ["CT", "MRI", "US", "X-ray"]:
                return self.pick("report_generation", modality, prefer=["LLaVA-Rad"])
            return None
        if chain_task == "clinical_summary":
            return self.pick("summarization", modality, prefer=["LLaVA"]) or self.pick("qa", modality, prefer=["PMC-LLaMA"])
        return None

# ===============================
# Planner (modality + chain + scenario)
# ===============================

class ScenarioPlanner:
    def __init__(self, registry: ToolRegistry, bank: "EnhancedSingleRoundBank"):
        self.registry = registry
        self.bank = bank
        self.router = Router(registry)

        # Modality sampling weights
        self.modality_weights = {
            "CT": 0.22,   # 略抬高 CT/MRI 以便多些注册类和多图
            "MRI": 0.22,
            "X-ray": 0.18,
            "US": 0.10,
            "Histology": 0.12,
            "Retina-Fundus": 0.06,
            "Retina-OCT": 0.04,
            "Gross": 0.03,
            "Cell-Microscopy": 0.03,
        }

    def sample_modality(self) -> str:
        items, w = zip(*self.modality_weights.items())
        return random.choices(items, weights=w, k=1)[0]

    def valid_chains_for_modality(self, modality: str) -> List[ChainType]:
        if modality in ["CT", "MRI"]:
            return [
                ChainType.COMPLETE_IMAGING,
                ChainType.COMPARATIVE,
                ChainType.DIAGNOSTIC,
                ChainType.REPORT_SUMMARY_QA,
                ChainType.ANALYSIS_QA,
                ChainType.SEGMENT_SUMMARY_QA,
                ChainType.RANDOM_MIX,  # 仅 CT/MRI 支持随机混合（含注册）
            ]
        if modality in ["X-ray", "US"]:
            return [
                ChainType.DIAGNOSTIC,
                ChainType.REPORT_SUMMARY_QA,
                ChainType.ANALYSIS_QA,
                ChainType.SEGMENT_SUMMARY_QA,
            ]
        if modality in ["Histology", "Retina-Fundus", "Retina-OCT", "Gross", "Cell-Microscopy"]:
            return [
                ChainType.DIAGNOSTIC,
                ChainType.REPORT_SUMMARY_QA,
                ChainType.ANALYSIS_QA,
                ChainType.SEGMENT_SUMMARY_QA,
            ]
        return [
            ChainType.DIAGNOSTIC,
            ChainType.REPORT_SUMMARY_QA,
            ChainType.ANALYSIS_QA,
        ]

    def valid_scenarios_for(self, chain: ChainType, modality: str) -> List[ConversationScenario]:
        if chain == ChainType.COMPLETE_IMAGING:
            return [ConversationScenario.SINGLE_IMAGE]
        if chain == ChainType.DIAGNOSTIC:
            # 更偏向中途换图，增加多图片占比
            return [ConversationScenario.SWITCH_IMAGE_MID, ConversationScenario.SINGLE_IMAGE]
        if chain == ChainType.COMPARATIVE:
            if modality in ["CT", "MRI"]:
                return [ConversationScenario.REGISTRATION_FROM_START, ConversationScenario.REGISTRATION_ADD_LATER]
            return []
        if chain == ChainType.RANDOM_MIX:
            # 先 QA 后 Registration，自然是“后加第二张图”
            return [ConversationScenario.REGISTRATION_ADD_LATER]
        return [ConversationScenario.SINGLE_IMAGE]

    def plan(self) -> Tuple[ChainType, ConversationScenario, str, List[str]]:
        # 1) Pick modality
        modality = self.sample_modality()

        # 2) Pick chain (短链 & RANDOM_MIX 概率更高，全流程略低)
        chains = self.valid_chains_for_modality(modality)
        weights = []
        for ch in chains:
            if ch in [ChainType.REPORT_SUMMARY_QA, ChainType.ANALYSIS_QA, ChainType.SEGMENT_SUMMARY_QA]:
                weights.append(2.0)   # 短链偏高
            elif ch == ChainType.RANDOM_MIX:
                weights.append(2.2)   # 随机混合更高，鼓励“跨任务 + 多图”
            elif ch == ChainType.COMPLETE_IMAGING:
                weights.append(0.7)   # 全流程稍降
            else:
                weights.append(1.0)
        chain_type = random.choices(chains, weights=weights, k=1)[0]

        # 3) Pick scenario (在 DIAGNOSTIC 里更倾向 SWITCH_IMAGE_MID)
        scenarios = self.valid_scenarios_for(chain_type, modality)
        if not scenarios:
            chain_type = ChainType.DIAGNOSTIC
            scenarios = [ConversationScenario.SWITCH_IMAGE_MID, ConversationScenario.SINGLE_IMAGE]
        # 加权：SWITCH_IMAGE_MID 更大概率
        if chain_type == ChainType.DIAGNOSTIC and len(scenarios) == 2:
            scenario = random.choices(scenarios, weights=[1.7, 1.0], k=1)[0]
        else:
            scenario = random.choice(scenarios)

        # 4) Convert chain tasks → tools via router
        tools: List[str] = []
        for task in CHAIN_TASKS[chain_type]:
            tool = self.router.route_chain_task(task, modality)
            if tool:
                tools.append(tool)

        # ensure at least 2 steps; otherwise fallback
        if len(tools) < 2:
            if modality in ["US"]:
                tools = ["UltraSAM", "LLaVA-Rad"]
            elif modality in ["CT", "MRI", "X-ray"]:
                tools = ["BiomedClip", "LLaVA-Rad"]
            elif modality in ["Histology", "Cell-Microscopy"]:
                tools = ["CONCH" if self.registry.has("CONCH") else "DSMIL", "PMC-LLaMA"]
            else:
                tools = ["BiomedClip", "PMC-LLaMA"]
            chain_type = ChainType.DIAGNOSTIC
            scenario = ConversationScenario.SWITCH_IMAGE_MID  # fallback 也倾向多图

        return chain_type, scenario, modality, tools

# ===============================
# Single-Round Example Bank
# ===============================

class EnhancedSingleRoundBank:
    def __init__(self, root: Union[str, Path]):
        self.extractor = RealDataExtractor(root)

    def get_example(self, tool_name: str) -> Optional[ToolExample]:
        return self.extractor.get_random_example(tool_name)

# ===============================
# Conversation Builder
# ===============================

class ScenarioAwareBuilder:
    def __init__(self, registry: ToolRegistry, bank: EnhancedSingleRoundBank):
        self.registry = registry
        self.bank = bank

    def _emit_image_tags(self, n: int) -> str:
        return "".join("<image>\n" for _ in range(max(0, n)))

    def build_conversation(self, chain_type: ChainType, scenario: ConversationScenario, modality: str, planned_tools: List[str]) -> Dict[str, Any]:
        state = ConvState(session_id=str(uuid.uuid4()), scenario=scenario, chain_type=chain_type, modality=modality)
        conversations: List[Dict[str, Any]] = []

        # Anchor first image via first tool example
        first_example = self.bank.get_example(planned_tools[0])
        if not first_example:
            return {}
        primary = self._primary_image_path(first_example.image_path)
        state.base_image_id = first_example.image_id
        state.base_image_path = primary

        state.all_image_paths = [primary] if primary else []
        state.has_second_image = False
        state.second_image_added_at_turn = -1
        state.image_switch_turn = -1

        # ===== 构造每一轮 =====
        for i, tool_name in enumerate(planned_tools):
            turn = self._build_turn(tool_name, state, i, planned_tools)
            if not turn:
                continue
            conversations.extend(turn)
            state.tool_history.append(tool_name)

        # ===== 校验 =====
        if not self._validate_conversation(conversations, state):
            return {}

        image_field: Union[str, List[str]] = state.all_image_paths[0] if len(state.all_image_paths) == 1 else state.all_image_paths
        return {"image": image_field, "conversations": conversations}

    # ---------- helpers ----------
    def _primary_image_path(self, image: Union[str, List[str], None]) -> Optional[str]:
        if isinstance(image, list):
            return image[0] if image else None
        return image

    def _add_image_to_state(self, state: ConvState, image_path: Union[str, List[str], None]) -> None:
        if not image_path:
            return
        if isinstance(image_path, list):
            for p in image_path:
                if isinstance(p, str) and p and p not in state.all_image_paths:
                    state.all_image_paths.append(p)
        else:
            if isinstance(image_path, str) and image_path and image_path not in state.all_image_paths:
                state.all_image_paths.append(image_path)

    def _build_turn(self, tool_name: str, state: ConvState, turn_idx: int, chain: List[str]) -> List[Dict[str, Any]]:
        ex = self.bank.get_example(tool_name)
        if not ex:
            return []
        user_prompt = self._adapt_user_prompt_for_scenario(ex, state, turn_idx, chain, tool_name)
        assistant_call = self._create_assistant_call(ex, state)
        tool_output = self._create_tool_output(ex.tool_name, ex.tool_output, ex.input_prompt)
        final_response = self._create_final_response(ex, state)
        self._update_state_with_artifacts(state, tool_name, ex)
        return [
            {"from": "human", "value": user_prompt},
            assistant_call,
            {"from": "human", "value": tool_output},
            final_response
        ]

    def _adapt_user_prompt_for_scenario(self, ex: ToolExample, state: ConvState,
                                        turn_idx: int, chain: List[str], tool_name: str) -> str:
        clean_prompt = ex.input_prompt.strip()
        tool = self.registry[tool_name]

        # -------- 首轮 --------
        if turn_idx == 0:
            if tool.requires_two_images:
                # 直接引入两张图（配准等）
                second = self.bank.extractor.get_different_image(state.all_image_paths)
                if second:
                    self._add_image_to_state(state, second)
                    state.has_second_image = True
                    state.second_image_added_at_turn = 0
                return f"{self._emit_image_tags(2)}{clean_prompt}"
            else:
                # 普通：引入首图
                return f"{self._emit_image_tags(1)}{clean_prompt}"

        # -------- 随机/指定的中途换图 --------
        if state.scenario == ConversationScenario.SWITCH_IMAGE_MID and state.image_switch_turn == -1:
            # 倾向在第 2 轮换图（或首次可换的轮次）
            if turn_idx in [1, 2]:
                diff = self.bank.extractor.get_different_image(state.all_image_paths)
                if diff:
                    self._add_image_to_state(state, diff)
                    state.image_switch_turn = turn_idx
                    return f"I now have a different image to analyze. {self._emit_image_tags(1)}{clean_prompt}"

        # -------- 迟加第二张图（注册类/RANDOM_MIX） --------
        if (tool.requires_two_images and not state.has_second_image) or \
           (state.scenario == ConversationScenario.REGISTRATION_ADD_LATER and not state.has_second_image and canonical_tool_name(tool_name) == "UniGradICON"):
            second = self.bank.extractor.get_different_image(state.all_image_paths)
            if second:
                self._add_image_to_state(state, second)
                state.has_second_image = True
                state.second_image_added_at_turn = turn_idx
                return f"Now I have a second image. {self._emit_image_tags(1)}Register this new scan with the previous one."

        # -------- 其余情况：复用已有图片，不打 tag --------
        return self._followup(clean_prompt)

    def _followup(self, text: str) -> str:
        opts = [
            f"Now, {text.lower()}",
            f"Following up on the previous analysis, {text.lower()}",
            f"Next, {text.lower()}",
            f"Building on the results so far, {text.lower()}",
            f"Using the previous output, {text.lower()}",
        ]
        return random.choice(opts)

    def _create_assistant_call(self, ex: ToolExample, state: ConvState) -> Dict[str, Any]:
        thoughts = ex.thoughts
        if state.tool_history:
            thoughts = f"Building on the previous {', '.join(state.tool_history)} results. {thoughts}"
        return {
            "from": "gpt",
            "thoughts": thoughts,
            "actions": [{
                "API_name": ex.tool_name,
                "API_params": ex.tool_params
            }],
            "value": f"I'll use {ex.tool_name} to complete this request based on our current analysis."
        }

    MAX_TOOL_OUTPUT_CHARS = 1200  # 轻量截断，防止后续步骤失败

    def _create_tool_output(self, tool_name: str, raw_output: str, original_prompt: str) -> str:
        s = raw_output or ""
        if len(s) > self.MAX_TOOL_OUTPUT_CHARS:
            s = s[:self.MAX_TOOL_OUTPUT_CHARS] + f"\n...[TRUNCATED {len(raw_output)-self.MAX_TOOL_OUTPUT_CHARS} chars]"
        return f"{tool_name} output: {s}\n\nAnswer my first request: {original_prompt}"

    def _create_final_response(self, ex: ToolExample, state: ConvState) -> Dict[str, Any]:
        thoughts = f"Based on the {ex.tool_name} output, I can now provide a comprehensive answer."
        if state.conversation_context:
            thoughts += " This builds on our previous analysis."
        return {"from": "gpt", "thoughts": thoughts, "actions": [], "value": ex.assistant_response}

    def _update_state_with_artifacts(self, state: ConvState, tool_name: str, ex: ToolExample) -> None:
        t = canonical_tool_name(tool_name)
        artifact_types = {
            "UniGradICON": "registered_image",
            "UltraSAM": "segmentation_mask",
            "HealthGPT": "reconstructed_image",
            "IterNet": "fundus_mask",
            "LLaVA-Rad": "radiology_report",
            "LLaVA": "summary_text",
            "RaTE-NER": "extracted_entities",
            "PMC-LLaMA": "qa_response",
            "SpecialistVLMs": "specialist_report",
            "LLaVA-Med": "medical_vqa_response",
            "BiomedClip": "medical_classification",
            "grounding dino": "object_grounding",
            "MedSAM": "medical_segmentation",
            "grounding dino + MedSAM": "grounded_segmentation",
            # "ChatCAD-G": "medical_report",
            "ChatCAD-R": "rag_medical_response",
            "CONCH": "tissue_classification",
            "DSMIL": "tumor_detection",
            "DSMIL-TCGA": "tumor_detection",
            "DSMIL-C16": "tumor_detection",
            "CellViT": "cell_segmentation",
            "CellSAM": "wsi_segmentation",
        }
        a_type = artifact_types.get(t, "output")
        artifact = Artifact(
            id=f"{t.lower()}_{len(state.artifacts):03d}",
            type=a_type,
            source_tool=t,
            content=ex.tool_output,
            metadata={"params": ex.tool_params}
        )
        state.add_artifact(artifact)
        state.add_context(f"{t} generated {a_type}")

    def _validate_conversation(self, conversations: List[Dict[str, Any]], state: ConvState) -> bool:
        # Count <image> tags
        tag_count = 0
        for conv in conversations:
            if conv.get("from") == "human":
                tag_count += conv.get("value", "").count("<image>")

        if tag_count != len(state.all_image_paths):
            print(f"Tag/image mismatch: {tag_count} tags vs {len(state.all_image_paths)} images")
            return False

        # Registration scenarios must be CT/MRI with >=2 images
        if state.scenario in [ConversationScenario.REGISTRATION_FROM_START, ConversationScenario.REGISTRATION_ADD_LATER]:
            if state.modality not in ["CT", "MRI"]:
                print("Registration scenarios only valid for CT/MRI.")
                return False
            if len(state.all_image_paths) < 2:
                print("Registration scenarios require 2 images.")
                return False

        # Pathology should not use radiology-only tools
        if state.modality in ["Histology", "Histology-WSI", "Histology-Patch", "Cell-Microscopy"]:
            forbidden = {"LLaVA-Rad", "UniGradICON"}
            if any(t in forbidden for t in state.tool_history):
                print("Pathology chain included radiology-only tools.")
                return False

        # at least 2 tool calls
        if len([c for c in conversations if c.get("from") == "gpt" and c.get("actions")]) < 2:
            print("Chain too short after filtering.")
            return False

        return True

# ===============================
# Builder function & CLI
# ===============================

def build_enhanced_conversation(registry: ToolRegistry, bank: EnhancedSingleRoundBank) -> Dict[str, Any]:
    planner = ScenarioPlanner(registry, bank)
    builder = ScenarioAwareBuilder(registry, bank)

    chain_type, scenario, modality, tools = planner.plan()
    try:
        convo = builder.build_conversation(chain_type, scenario, modality, tools)
        return convo
    except Exception as e:
        print(f"Error building {scenario.value} conversation: {e}")
        return {}

def infer_scenario_from_conversation(conversation: Dict[str, Any]) -> ConversationScenario:
    image_field = conversation.get("image")
    conversations = conversation.get("conversations", [])
    image_tag_pattern = []
    for conv in conversations:
        if conv.get("from") == "human":
            c = conv.get("value", "").count("<image>")
            if c > 0:
                image_tag_pattern.append(c)
    if isinstance(image_field, list) and len(image_field) >= 2:
        if image_tag_pattern and image_tag_pattern[0] >= 2:
            return ConversationScenario.REGISTRATION_FROM_START
        elif len(image_tag_pattern) >= 2 and any(tags > 0 for tags in image_tag_pattern[1:]):
            for i, conv in enumerate(conversations):
                if conv.get("from") == "human" and i > 0:
                    v = conv.get("value", "").lower()
                    if ("second image" in v and "<image>" in conv.get("value", "")):
                        return ConversationScenario.REGISTRATION_ADD_LATER
                    if ("different image" in v and "<image>" in conv.get("value", "")):
                        return ConversationScenario.SWITCH_IMAGE_MID
    return ConversationScenario.SINGLE_IMAGE

def main():
    parser = argparse.ArgumentParser(description="Generate modality-aware multi-tool multi-round dialogues")
    parser.add_argument("--tool_yaml", type=str, required=True, help="Path to tool metadata YAML")
    parser.add_argument("--single_round_dir", type=str, required=True, help="Path to single-round examples directory")
    parser.add_argument("--out", type=str, required=True, help="Output file path (.jsonl)")
    parser.add_argument("--num", type=int, required=True, help="Number of conversations to generate")
    args = parser.parse_args()

    registry = ToolRegistry(args.tool_yaml)
    bank = EnhancedSingleRoundBank(args.single_round_dir)

    print("Loading tool examples...")
    total = 0
    for tool in registry.tools:
        exs = bank.extractor.load_tool_examples(tool)
        print(f"{tool}: loaded {len(exs)} examples")
        total += len(exs)
    print(f"Total examples loaded: {total}")
    print(f"Image pool size: {len(bank.extractor._image_pool)}")

    conversations = []
    scenario_counts = {s: 0 for s in ConversationScenario}
    successful = 0
    skipped = 0

    print(f"Generating {args.num} conversations...")
    for i in range(args.num):
        conv = build_enhanced_conversation(registry, bank)
        if conv and conv.get("conversations"):
            conversations.append(conv)
            successful += 1
            scenario_counts[infer_scenario_from_conversation(conv)] += 1
        else:
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"Progress {i+1}/{args.num} | Success {successful} | Skipped {skipped}")

    print("\nFinal Results:")
    print(f"Generated: {len(conversations)} valid")
    print(f"Success rate: {len(conversations)/max(args.num,1)*100:.1f}%")
    print(f"Total skipped: {skipped}")
    print("\nScenario Distribution:")
    for s, c in scenario_counts.items():
        pct = (c / len(conversations) * 100) if conversations else 0
        print(f"  {s.value}: {c} ({pct:.1f}%)")

    with open(args.out, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(conversations)} conversations to {args.out}")

if __name__ == "__main__":
    main()
