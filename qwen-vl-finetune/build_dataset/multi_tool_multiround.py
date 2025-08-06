"""
python3 multi_tool_multiround.py \
--tool_yaml corpus_pack/tool_meta.yaml \
--single_round_dir tool_instruct \
--out multi_round/multi_tool_multiround.jsonl \
--num 100000

{
  "image": …,
  "conversations": [ {from, value, thoughts, actions, …}, … ]
}
"""

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

# -----------------------------------------------------------------------------
# Scenario Definitions ---------------------------------------------------------
# -----------------------------------------------------------------------------

class ConversationScenario(Enum):
    SINGLE_IMAGE = "single_image"  # Scenario 1: One image throughout
    REGISTRATION_FROM_START = "registration_from_start"  # Scenario 2: Two images from start
    REGISTRATION_ADD_LATER = "registration_add_later"  # Scenario 3: Add second image later
    SWITCH_IMAGE_MID = "switch_image_mid"  # Scenario 4: Switch to different image

# -----------------------------------------------------------------------------
# Tool metadata ----------------------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    modality: str
    refine_task: str
    default_args: Dict[str, Any]
    refine_responses: List[str]
    success_responses: List[str]

    @classmethod
    def from_dict(cls, name: str, cfg: Dict[str, Any]) -> "Tool":
        return cls(
            name=name,
            modality=cfg.get("modality", ""),
            refine_task=cfg.get("refine_task", ""),
            default_args=cfg.get("default_args", {}),
            refine_responses=cfg.get("refine_responses", []),
            success_responses=cfg.get("success_responses", []),
        )


class ToolRegistry:
    """Enhanced wrapper for tool-metadata lookup with successor logic."""

    def __init__(self, yaml_path: Union[str, Path]):
        with Path(yaml_path).open() as f:
            raw = yaml.safe_load(f)
        self.tools: Dict[str, Tool] = {name: Tool.from_dict(name, cfg) for name, cfg in raw.items()}
        self._successors = self._build_successor_map()

    def _build_successor_map(self) -> Dict[str, List[str]]:
        """Build logical tool succession mapping based on medical workflow."""
        mapping: Dict[str, List[str]] = {name: [] for name in self.tools}
        
        # Define logical successors based on medical workflow
        workflow_chains = {
            "UniGradICON": ["UltraSAM", "LLaVA-Rad", "HealthGPT"],
            "UltraSAM": ["LLaVA-Rad", "SpecialistVLMs"],
            "HealthGPT": ["LLaVA-Rad", "UltraSAM"],
            "IterNet": ["SpecialistVLMs", "LLaVA-Rad"],
            "LLaVA-Rad": ["LLaVA", "PMC-LLaMA"],
            "LLaVA": ["RaTE-NER", "PMC-LLaMA"],
            "RaTE-NER": ["PMC-LLaMA"],
            "PMC-LLaMA": ["LLaVA"],
            "SpecialistVLMs": ["LLaVA", "PMC-LLaMA"],
            "LLaVA-Med": ["PMC-LLaMA", "ChatCAD-G", "BiomedClip", "ChatCAD-R"],
            "BiomedClip": ["LLaVA-Med", "grounding dino", "ChatCAD-G", "SpecialistVLMs"],
            "grounding dino": ["MedSAM", "grounding dino + MedSAM", "LLaVA-Med", "ChatCAD"],
            "MedSAM": ["LLaVA-Med", "ChatCAD-G", "SpecialistVLMs", "BiomedClip"],
            "grounding dino + MedSAM": ["LLaVA-Med", "ChatCAD-G", "MedSAM", "SpecialistVLMs"],
            "ChatCAD-G": ["ChatCAD-R", "LLaVA-Med", "PMC-LLaMA"],
            "ChatCAD-R": ["LLaVA-Med", "PMC-LLaMA", "ChatCAD-G"],
        }
        
        # Add workflow successors
        for tool, successors in workflow_chains.items():
            mapping[tool].extend(successors)
        
        return mapping

    def successors(self, name: str) -> List[str]:
        return self._successors.get(name, [])

    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]


# -----------------------------------------------------------------------------
# Enhanced conversation state --------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class Artifact:
    """Represents a generated artifact that can be referenced later."""
    id: str
    type: str
    source_tool: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvState:
    """Enhanced conversation state with scenario tracking."""
    session_id: str
    scenario: ConversationScenario
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
        """Add a new artifact to the state."""
        self.artifacts[artifact.id] = artifact
        
    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID."""
        return self.artifacts.get(artifact_id)
        
    def get_artifacts_by_type(self, artifact_type: str) -> List[Artifact]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts.values() if a.type == artifact_type]
    
    def add_context(self, context: str) -> None:
        """Add context information for conversation continuity."""
        if context:
            self.conversation_context.append(context)


# -----------------------------------------------------------------------------
# Real data extraction and adaptation -----------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class ToolExample:
    """Represents a real tool usage example extracted from datasets."""
    tool_name: str
    image_id: str
    image_path: str
    input_prompt: str
    tool_params: Dict[str, Any]
    tool_output: str
    assistant_response: str
    thoughts: str


class RealDataExtractor:
    """Extracts and adapts real tool examples from the datasets."""
    
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
            "DSMIL": "DSMIL.jsonl",
            "CellViT": "CellViT.jsonl",
            "CellSAM": "CellSAM.jsonl",
            "LLaVA-Med": "LLaVA-Med.jsonl",
            "BiomedClip": "BiomedClip.jsonl",
            "grounding dino": "GD.jsonl",
            "MedSAM": "MedSAM.jsonl",
            "grounding dino + MedSAM": "GD_MedSAM.jsonl",
            "ChatCAD-G": "ChatCAD-G.jsonl",
            "ChatCAD-R": "ChatCAD-R.jsonl",
        }
        self._cache: Dict[str, List[ToolExample]] = {}
        self._image_pool: List[str] = []
        self._load_image_pool()
        
    def _load_image_pool(self) -> None:
        """Load a pool of different image paths for multi-image scenarios."""
        try:
            for tool_name in list(self.tool_mapping.keys())[:5]:  # Load from first 5 tools
                examples = self.load_tool_examples(tool_name, max_examples=50)
                for example in examples:
                    if example.image_path and example.image_path not in self._image_pool:
                        self._image_pool.append(example.image_path)
        except Exception as e:
            print(f"Warning: Could not load image pool: {e}")
    
    def get_different_image(self, exclude_paths: List[str]) -> Optional[str]:
        """Get a different image path from the pool."""
        available = [img for img in self._image_pool if img not in exclude_paths]
        return random.choice(available) if available else None
        
    def load_tool_examples(self, tool_name: str, max_examples: int = 100000) -> List[ToolExample]:
        """Load real examples for a specific tool."""
        if tool_name in self._cache:
            return self._cache[tool_name]
            
        if tool_name not in self.tool_mapping:
            return []
            
        dataset_file = self.tool_instruct_dir / self.tool_mapping[tool_name]
        if not dataset_file.exists():
            return []
            
        examples = []
        try:
            with dataset_file.open() as f:
                for i, line in enumerate(f):
                    if i >= max_examples:
                        break
                    data = json.loads(line.strip())
                    example = self._parse_example(tool_name, data)
                    if example:
                        examples.append(example)
        except Exception as e:
            print(f"Error loading {tool_name} examples: {e}")
            
        self._cache[tool_name] = examples
        return examples
    
    def _parse_example(self, tool_name: str, data: Dict[str, Any]) -> Optional[ToolExample]:
        """Parse a single example from the dataset."""
        conversations = data.get("conversations", [])
        if len(conversations) < 3:
            return None
        
        assistant_call = conversations[1]
        actions = assistant_call.get("actions", [])
        if not actions:
            return None
            
        actual_tool_name = actions[0].get("API_name", "")
        if actual_tool_name != tool_name:
            return None
            
        # FIXED: Clean all image tags from input prompt
        user_prompt = conversations[0]["value"]
        user_prompt = user_prompt.replace("<image>\n", "").replace("<image>", "").strip()
        
        thoughts = assistant_call.get("thoughts", "")
        tool_params = actions[0]["API_params"] if actions else {}
        
        tool_output_msg = conversations[2]["value"]
        tool_output = tool_output_msg.split("Answer my first request:")[0].strip()
        if tool_output.startswith(f"{tool_name} output:"):
            tool_output = tool_output[len(f"{tool_name} output:"):].strip()
            
        assistant_response = conversations[3]["value"] if len(conversations) > 3 else ""
        
        # Handle potentially nested image paths
        image_data = data.get("image") or data.get("images")
        if isinstance(image_data, list):
            image_path = image_data[0] if image_data else None
        else:
            image_path = image_data
        
        return ToolExample(
            tool_name=tool_name,
            image_id=data.get("image_id") or " ",
            image_path=image_path,
            input_prompt=user_prompt,
            tool_params=tool_params,
            tool_output=tool_output,
            assistant_response=assistant_response,
            thoughts=thoughts
        )
    
    def get_random_example(self, tool_name: str) -> Optional[ToolExample]:
        """Get a random example for a tool."""
        examples = self.load_tool_examples(tool_name)
        return random.choice(examples) if examples else None


# -----------------------------------------------------------------------------
# Enhanced single-round example bank ------------------------------------------
# -----------------------------------------------------------------------------

class EnhancedSingleRoundBank:
    """Enhanced bank that works with real data extractor."""

    def __init__(self, root: Union[str, Path]):
        self.extractor = RealDataExtractor(root)

    def get_example(self, tool_name: str) -> Optional[ToolExample]:
        """Get a real example for the specified tool."""
        return self.extractor.get_random_example(tool_name)


# -----------------------------------------------------------------------------
# Scenario-aware conversation planning ----------------------------------------
# -----------------------------------------------------------------------------

class ScenarioPlanner:
    """Plans conversation chains based on specific scenarios."""
    
    def __init__(self, registry: ToolRegistry, bank: EnhancedSingleRoundBank):
        self.registry = registry
        self.bank = bank
        
        # Scenario probabilities
        self.scenario_weights = {
            ConversationScenario.SINGLE_IMAGE: 0.5,
            ConversationScenario.REGISTRATION_FROM_START: 0.2,
            ConversationScenario.REGISTRATION_ADD_LATER: 0.2,
            ConversationScenario.SWITCH_IMAGE_MID: 0.1
        }
        
        # Tool chains for different scenarios
        self.scenario_chains = {
            ConversationScenario.SINGLE_IMAGE: [
                ["UltraSAM", "LLaVA-Rad", "LLaVA", "RaTE-NER"],  # Complete imaging pipeline
                ["BiomedClip", "LLaVA-Med", "ChatCAD-G"],  # Classification -> VQA -> Report
                ["grounding dino", "MedSAM", "ChatCAD-G", "PMC-LLaMA"],  # Detection workflow
                ["HealthGPT", "SpecialistVLMs", "LLaVA"],  # Enhancement workflow
            ],
            ConversationScenario.REGISTRATION_FROM_START: [
                ["UniGradICON", "UltraSAM", "LLaVA-Rad"],  # Registration first
                ["UniGradICON", "BiomedClip", "ChatCAD-G"],  # Registration -> analysis
            ],
            ConversationScenario.REGISTRATION_ADD_LATER: [
                ["UltraSAM", "LLaVA-Rad", "UniGradICON", "ChatCAD-G"],  # Add registration later
                ["BiomedClip", "LLaVA-Med", "UniGradICON", "PMC-LLaMA"],  # Add registration mid-chain
            ],
            ConversationScenario.SWITCH_IMAGE_MID: [
                ["UltraSAM", "BiomedClip", "LLaVA-Med", "ChatCAD-G"],  # Switch in middle
                ["grounding dino", "HealthGPT", "SpecialistVLMs"],  # Switch after first two
            ]
        }
        
    def plan_conversation(self) -> Tuple[List[str], ConversationScenario]:
        """Plan a conversation with a specific scenario."""
        # Select scenario
        scenario = random.choices(
            list(self.scenario_weights.keys()),
            weights=list(self.scenario_weights.values())
        )[0]
        
        # Select chain for scenario
        chains = self.scenario_chains[scenario]
        base_chain = random.choice(chains)
        
        # Verify tools have available data
        available_chain = []
        for tool in base_chain:
            example = self.bank.get_example(tool)
            if example:
                available_chain.append(tool)
        
        # Ensure minimum chain length
        if len(available_chain) < 2:
            # Fallback to single image scenario
            scenario = ConversationScenario.SINGLE_IMAGE
            available_chain = ["UltraSAM", "LLaVA-Rad"]  # Simple fallback
                
        return available_chain, scenario


# -----------------------------------------------------------------------------
# Scenario-aware conversation building -----------------------------------------
# -----------------------------------------------------------------------------

class ScenarioAwareBuilder:
    """Builds conversations according to specific scenarios."""
    
    def __init__(self, registry: ToolRegistry, bank: EnhancedSingleRoundBank):
        self.registry = registry
        self.bank = bank
        
    def build_conversation(self, planned_chain: List[str], scenario: ConversationScenario) -> Dict[str, Any]:
        """Build a conversation for a specific scenario."""
        if not planned_chain:
            return {}
            
        # Initialize conversation state with scenario
        state = ConvState(session_id=str(uuid.uuid4()), scenario=scenario)
        conversations = []
        
        # Get base image from first tool
        first_example = self.bank.get_example(planned_chain[0])
        if not first_example:
            return {}
            
        state.base_image_id = first_example.image_id
        state.base_image_path = first_example.image_path
        self._add_image_to_state(state, first_example.image_path)
        
        # Handle scenario-specific initialization
        if scenario == ConversationScenario.REGISTRATION_FROM_START:
            # Add second image immediately for registration
            second_image = self.bank.extractor.get_different_image(state.all_image_paths)
            if second_image:
                self._add_image_to_state(state, second_image)
                state.has_second_image = True
        
        # Build each conversation turn
        for i, tool_name in enumerate(planned_chain):
            turn = self._build_turn(tool_name, state, i, planned_chain)
            if turn:
                conversations.extend(turn)
                state.tool_history.append(tool_name)
        
        # Final validation
        if not self._validate_conversation(conversations, state):
            print(f"Warning: {scenario.value} conversation failed validation, skipping...")
            return {}
        
        # Prepare output
        if len(state.all_image_paths) > 1:
            image_field = state.all_image_paths
        else:
            image_field = state.all_image_paths[0] if state.all_image_paths else state.base_image_path
        
        return {
            "session_id": state.session_id,
            "image_id": state.base_image_id,
            "image": image_field,
            "file_name": state.base_image_path,
            "conversations": conversations
        }
    
    def _add_image_to_state(self, state: ConvState, image_path: Any) -> None:
        """Add image paths to state, ensuring no duplicates or nesting."""
        if isinstance(image_path, list):
            for img in image_path:
                if isinstance(img, str) and img and img not in state.all_image_paths:
                    state.all_image_paths.append(img)
        elif isinstance(image_path, str) and image_path and image_path not in state.all_image_paths:
            state.all_image_paths.append(image_path)
    
    def _validate_conversation(self, conversations: List[Dict[str, Any]], state: ConvState) -> bool:
        """Validate that conversation follows scenario requirements."""
        # Count image tags
        total_tags = 0
        for conv in conversations:
            if conv.get("from") == "human" and "<image>" in conv.get("value", ""):
                total_tags += conv["value"].count("<image>")
        
        expected_images = len(state.all_image_paths)
        
        if total_tags != expected_images:
            print(f"Tag/image mismatch: {total_tags} tags vs {expected_images} images")
            return False
        
        # Scenario-specific validation
        if state.scenario == ConversationScenario.REGISTRATION_FROM_START and expected_images < 2:
            print("Registration from start scenario needs 2+ images")
            return False
            
        return True
    
    def _build_turn(self, tool_name: str, state: ConvState, turn_idx: int, chain: List[str]) -> List[Dict[str, Any]]:
        """Build a single conversation turn based on scenario."""
        example = self.bank.get_example(tool_name)
        if not example:
            return []
            
        # Adapt user prompt based on scenario
        user_prompt = self._adapt_user_prompt_for_scenario(example, state, turn_idx, chain, tool_name)
        
        # Create assistant tool call
        assistant_call = self._create_assistant_call(example, state)
        
        # Create tool output
        tool_output = self._create_tool_output(example, state, tool_name)
        
        # Create final response
        final_response = self._create_final_response(example, state)
        
        # Update state
        self._update_state_with_artifacts(state, tool_name, example)
        
        return [
            {"from": "human", "value": user_prompt},
            assistant_call,
            {"from": "human", "value": tool_output},
            final_response
        ]
    
    def _adapt_user_prompt_for_scenario(self, example: ToolExample, state: ConvState, turn_idx: int, chain: List[str], tool_name: str) -> str:
        """Adapt user prompt based on specific scenario requirements."""
        clean_prompt = example.input_prompt.strip()
        
        # Scenario 1: Single Image Throughout
        if state.scenario == ConversationScenario.SINGLE_IMAGE:
            if turn_idx == 0:
                return f"<image>\n{clean_prompt}"
            else:
                return self._create_followup_prompt(clean_prompt)
        
        # Scenario 2: Registration with Two Images from Start
        elif state.scenario == ConversationScenario.REGISTRATION_FROM_START:
            if turn_idx == 0:
                if tool_name == "UniGradICON":
                    return f"<image>\n<image>\n{clean_prompt}"
                else:
                    # Non-registration tool but in registration scenario
                    return f"<image>\n<image>\n{clean_prompt}"
            else:
                return self._create_followup_prompt(clean_prompt)
        
        # Scenario 3: Registration - Add Second Image Later
        elif state.scenario == ConversationScenario.REGISTRATION_ADD_LATER:
            if turn_idx == 0:
                return f"<image>\n{clean_prompt}"
            elif tool_name == "UniGradICON" and not state.has_second_image:
                # This is where we add the second image
                second_image = self.bank.extractor.get_different_image(state.all_image_paths)
                if second_image:
                    self._add_image_to_state(state, second_image)
                    state.has_second_image = True
                    state.second_image_added_at_turn = turn_idx
                    return f"Now I have a second image. <image>\nRegister this new scan with the previous one."
                else:
                    return self._create_followup_prompt(clean_prompt)
            else:
                return self._create_followup_prompt(clean_prompt)
        
        # Scenario 4: Switch to Different Image Mid-Conversation
        elif state.scenario == ConversationScenario.SWITCH_IMAGE_MID:
            if turn_idx == 0:
                return f"<image>\n{clean_prompt}"
            elif turn_idx == 2 and state.image_switch_turn == -1:  # Switch at turn 2
                different_image = self.bank.extractor.get_different_image(state.all_image_paths)
                if different_image:
                    self._add_image_to_state(state, different_image)
                    state.image_switch_turn = turn_idx
                    return f"Now I have a different image to analyze. <image>\n{clean_prompt}"
                else:
                    return self._create_followup_prompt(clean_prompt)
            else:
                return self._create_followup_prompt(clean_prompt)
        
        # Fallback
        return f"<image>\n{clean_prompt}" if turn_idx == 0 else self._create_followup_prompt(clean_prompt)
    
    def _create_followup_prompt(self, clean_prompt: str) -> str:
        """Create follow-up prompt without new image tags."""
        followup_phrases = [
            f"Now, {clean_prompt.lower()}",
            f"Following up on the previous analysis, {clean_prompt.lower()}",
            f"Next, {clean_prompt.lower()}",
            f"Building on the results so far, {clean_prompt.lower()}",
            f"Using the previous output, {clean_prompt.lower()}"
        ]
        return random.choice(followup_phrases)
    
    def _create_assistant_call(self, example: ToolExample, state: ConvState) -> Dict[str, Any]:
        """Create assistant tool call response."""
        thoughts = example.thoughts
        if state.tool_history:
            thoughts = f"Building on the previous {', '.join(state.tool_history)} results. {thoughts}"
            
        return {
            "from": "gpt",
            "thoughts": thoughts,
            "actions": [{
                "API_name": example.tool_name,
                "API_params": example.tool_params
            }],
            "value": f"I'll use {example.tool_name} to complete this request based on our current analysis."
        }
    
    def _create_tool_output(self, example: ToolExample, state: ConvState, tool_name: str) -> str:
        """Create tool output message."""
        return f"{tool_name} output: {example.tool_output}\n\nAnswer my first request: {example.input_prompt}"
    
    def _create_final_response(self, example: ToolExample, state: ConvState) -> Dict[str, Any]:
        """Create final assistant response."""
        thoughts = f"Based on the {example.tool_name} output, I can now provide a comprehensive answer."
        if state.conversation_context:
            thoughts += f" This builds on our previous analysis."
            
        return {
            "from": "gpt", 
            "thoughts": thoughts,
            "actions": [],
            "value": example.assistant_response
        }
    
    def _update_state_with_artifacts(self, state: ConvState, tool_name: str, example: ToolExample) -> None:
        """Update conversation state with new artifacts."""
        artifact_id = f"{tool_name.lower()}_{len(state.artifacts):03d}"
        
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
            "CONCH": "tissue_classification",
            "DSMIL": "tumor_detection",
            "CellViT": "cell_segmentation",
            "CellSAM": "wsi_segmentation",
            "LLaVA-Med": "medical_vqa_response",
            "BiomedClip": "medical_classification",
            "grounding dino": "object_grounding",
            "MedSAM": "medical_segmentation",
            "grounding dino + MedSAM": "grounded_segmentation",
            "ChatCAD-G": "medical_report",
            "ChatCAD-R": "rag_medical_response",
        }
        
        artifact_type = artifact_types.get(tool_name, "output")
        artifact = Artifact(
            id=artifact_id,
            type=artifact_type,
            source_tool=tool_name,
            content=example.tool_output,
            metadata={"params": example.tool_params}
        )
        
        state.add_artifact(artifact)
        state.add_context(f"{tool_name} generated {artifact_type}")


# -----------------------------------------------------------------------------
# Main builder function -------------------------------------------------------
# -----------------------------------------------------------------------------

def build_enhanced_conversation(
    registry: ToolRegistry,
    bank: EnhancedSingleRoundBank
) -> Dict[str, Any]:
    """Build a single enhanced conversation with scenario awareness."""
    planner = ScenarioPlanner(registry, bank)
    builder = ScenarioAwareBuilder(registry, bank)
    
    # Plan conversation with scenario
    chain, scenario = planner.plan_conversation()
    if not chain:
        return {}
        
    # Build conversation for scenario
    try:
        conversation = builder.build_conversation(chain, scenario)
        return conversation
    except Exception as e:
        print(f"Error building {scenario.value} conversation: {e}")
        return {}


# -----------------------------------------------------------------------------
# CLI interface ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate scenario-based multi-round dialogues")
    parser.add_argument("--tool_yaml", type=str, required=True, help="Path to tool metadata YAML")
    parser.add_argument("--single_round_dir", type=str, required=True, help="Path to single-round examples directory")
    parser.add_argument("--out", type=str, required=True, help="Output file path")
    parser.add_argument("--num", type=int, required=True, help="Number of conversations to generate")
    args = parser.parse_args()

    registry = ToolRegistry(args.tool_yaml)
    bank = EnhancedSingleRoundBank(args.single_round_dir)

    # Check available examples for each tool
    print("Loading tool examples...")
    total_examples = 0
    for tool in registry.tools:
        exs = bank.extractor.load_tool_examples(tool)
        print(f"{tool}: loaded {len(exs)} examples")
        total_examples += len(exs)
    
    print(f"Total examples loaded: {total_examples}")
    print(f"Image pool size: {len(bank.extractor._image_pool)}")

    conversations = []
    scenario_counts = {scenario: 0 for scenario in ConversationScenario}
    successful = 0
    skipped = 0
    
    print(f"Generating {args.num} conversations across all scenarios...")
    
    for i in range(args.num):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{args.num} | Success: {successful} | Skipped: {skipped}")
            for scenario, count in scenario_counts.items():
                print(f"  {scenario.value}: {count}")
            
        conversation = build_enhanced_conversation(registry, bank)
        if conversation and conversation.get("conversations"):
            conversations.append(conversation)
            successful += 1
            
            # Track scenario distribution
            # Infer scenario from conversation structure
            scenario = infer_scenario_from_conversation(conversation)
            scenario_counts[scenario] += 1
        else:
            skipped += 1
            
        # Early warning if skip rate is too high
        if i > 200 and skipped > successful:
            print(f"Warning: High skip rate detected. Consider checking data availability.")

    print(f"\nFinal Results:")
    print(f"Generated: {len(conversations)} valid conversations")
    print(f"Success rate: {len(conversations)/args.num*100:.1f}%")
    print(f"Total skipped: {skipped}")
    
    print(f"\nScenario Distribution:")
    for scenario, count in scenario_counts.items():
        percentage = (count / len(conversations) * 100) if conversations else 0
        print(f"  {scenario.value}: {count} ({percentage:.1f}%)")

    # Write output
    with open(args.out, "w", encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(conversations)} conversations to {args.out}")


def infer_scenario_from_conversation(conversation: Dict[str, Any]) -> ConversationScenario:
    """Infer the scenario type from the conversation structure."""
    image_field = conversation.get("image")
    conversations = conversation.get("conversations", [])
    
    # Count image tags across all turns
    image_tag_pattern = []
    for conv in conversations:
        if conv.get("from") == "human":
            tag_count = conv.get("value", "").count("<image>")
            if tag_count > 0:
                image_tag_pattern.append(tag_count)
    
    # Determine scenario based on patterns
    if isinstance(image_field, list) and len(image_field) >= 2:
        if len(image_tag_pattern) >= 1 and image_tag_pattern[0] >= 2:
            return ConversationScenario.REGISTRATION_FROM_START
        elif len(image_tag_pattern) >= 2 and any(tags > 0 for tags in image_tag_pattern[1:]):
            # Check if second image was added later
            for i, conv in enumerate(conversations):
                if conv.get("from") == "human" and i > 0:
                    value = conv.get("value", "")
                    if ("second image" in value.lower() or "new image" in value.lower()) and "<image>" in value:
                        return ConversationScenario.REGISTRATION_ADD_LATER
                    elif ("different image" in value.lower() or "switch" in value.lower()) and "<image>" in value:
                        return ConversationScenario.SWITCH_IMAGE_MID
    
    # Default to single image scenario
    return ConversationScenario.SINGLE_IMAGE


if __name__ == "__main__":
    main()
