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

import yaml

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
            "UniGradICON": ["UltraSAM", "LLaVA-Rad", "HealthGPT"],  # Registration → Analysis
            "UltraSAM": ["LLaVA-Rad", "SpecialistVLMs"],  # Segmentation → Report
            "HealthGPT": ["LLaVA-Rad", "UltraSAM"],  # Reconstruction → Analysis
            "IterNet": ["SpecialistVLMs", "LLaVA-Rad"],  # Fundus → Specialist analysis
            "LLaVA-Rad": ["LLaVA", "PMC-LLaMA"],  # Report → Summary/QA
            "LLaVA": ["RaTE-NER", "PMC-LLaMA"],  # Summary → NER/QA
            "RaTE-NER": ["PMC-LLaMA"],  # NER → QA
            "PMC-LLaMA": ["LLaVA"],  # QA → Summary
            "SpecialistVLMs": ["LLaVA", "PMC-LLaMA"],  # Specialist → Summary/QA
            "LLaVA-Med": ["PMC-LLaMA", "ChatCAD", "BiomedCLIP", "ChatCAD+"],  # VQA -> QA/Report/Classification
            "BiomedCLIP": ["LLaVA-Med", "Grounding-DINO", "ChatCAD", "SpecialistVLMs"],  # Classification -> VQA/Grounding/Report
            "Grounding-DINO": ["MedSAM", "G-Seg", "LLaVA-Med", "ChatCAD"],  # Grounding -> Segmentation/VQA/Report
            "MedSAM": ["LLaVA-Med", "ChatCAD", "SpecialistVLMs", "BiomedCLIP"],  # Segmentation -> VQA/Report/Analysis
            "G-Seg": ["LLaVA-Med", "ChatCAD", "MedSAM", "SpecialistVLMs"],  # Combined G+Seg -> VQA/Report
            "ChatCAD": ["ChatCAD+", "LLaVA-Med", "PMC-LLaMA"],  # Report -> RAG/VQA/QA
            "ChatCAD+": ["LLaVA-Med", "PMC-LLaMA", "ChatCAD"],  # RAG -> VQA/QA/Report
        }
        
        # Add workflow successors
        for tool, successors in workflow_chains.items():
            if tool in mapping:  # Only add if tool exists in registry
                mapping[tool].extend([s for s in successors if s in self.tools])
        
        # Add some random cross-modal possibilities
        names = list(self.tools.keys())
        for tool in names:
            # Add 2-3 random other tools as potential successors
            others = [t for t in names if t != tool and t not in mapping[tool]]
            if others:
                mapping[tool].extend(random.sample(others, min(3, len(others))))
        
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
    type: str  # "image", "mask", "report", "summary", "entities", etc.
    source_tool: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvState:
    """Enhanced conversation state with artifact tracking."""
    session_id: str
    all_image_paths: List[str] = field(default_factory=list)
    has_second_image: bool = False
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    conversation_context: List[str] = field(default_factory=list)
    tool_history: List[str] = field(default_factory=list)
    current_modality: Optional[str] = None
    
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
    image_path: str
    input_prompt: str
    tool_params: Dict[str, Any]
    tool_output: str
    assistant_response: str
    thoughts: str
    modality: Optional[str] = None


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
            "conch": "instruction_2000_img_updated_conch.jsonl",
            "dsmil": "instruction_2000_img_updated_conch.jsonl",
            "Cellvit": "instruction_2000_img_updated_conch.jsonl",
            "cellsam": "instruction_2000_img_updated_conch.jsonl",
            "LLaVA-Med": "instruct_all.jsonl",           # VQA on medical images
            "BiomedCLIP": "instruct_all.jsonl", # Medical image classification
            "Grounding-DINO": "instruct_all.jsonl",     # Object grounding in medical images
            "MedSAM": "instruct_all.jsonl",        # Medical segmentation with bbox
            "G-Seg": "instruct_all.jsonl",               # Combined grounding + segmentation
            "ChatCAD": "instruct_all.jsonl",            # Medical report generation
            "ChatCAD+": "instruct_all.jsonl",         # RAG for medical queries
        }

        # Modality mapping for tools
        self.tool_modalities = {
            "LLaVA-Med": ["MRI", "CT", "X-ray", "Histology", "Gross"],
            "BiomedCLIP": ["MRI", "CT", "X-ray", "Histology", "Gross"],
            "Grounding-DINO": ["MRI", "CT", "X-ray", "Histology"],
            "MedSAM": ["MRI", "CT", "X-ray", "Histology", "Gross"],
            "G-Seg": ["MRI", "CT", "X-ray", "Histology"],
            "ChatCAD": ["X-ray"],
            "ChatCAD+": ["Any"],  # Can work with or without images
        }

        self._cache: Dict[str, List[ToolExample]] = {}
        
    def load_tool_examples(self, tool_name: str, max_examples: int = 100) -> List[ToolExample]:
        """Load real examples for a specific tool."""
        if tool_name in self._cache:
            return self._cache[tool_name]
            
        if tool_name not in self.tool_mapping:
            return []
            
        dataset_file = self.tool_instruct_dir / self.tool_mapping[tool_name]
        if not dataset_file.exists():
            return []
            
        examples = []
        with dataset_file.open() as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                data = json.loads(line.strip())
                example = self._parse_example(tool_name, data)
                if example:
                    examples.append(example)
            
        self._cache[tool_name] = examples
        return examples
    
    def _parse_example(self, tool_name: str, data: Dict[str, Any]) -> Optional[ToolExample]:
        """Parse a single example from the dataset."""
        conversations = data.get("conversations", [])
        if len(conversations) < 3:  # Need at least user → assistant → user → assistant
            return None
        
        # Extract assistant tool call to check if it matches the requested tool
        assistant_call = conversations[1]
        actions = assistant_call.get("actions", [])
        if not actions:
            return None
            
        # Check if this example is for the requested tool
        actual_tool_name = actions[0].get("API_name", "")
        if actual_tool_name != tool_name:
            return None  # Skip examples not for this specific tool
            
        # Extract user prompt (first human message)
        user_prompt = conversations[0]["value"].replace("<image>\n", "").strip()
        
        thoughts = assistant_call.get("thoughts", "")
        tool_params = actions[0]["API_params"] if actions else {}
        
        # Extract tool output (second human message contains tool output)
        tool_output_msg = conversations[2]["value"]
        tool_output = tool_output_msg.split("Answer my first request:")[0].strip()
        if tool_output.startswith(f"{tool_name} output:"):
            tool_output = tool_output[len(f"{tool_name} output:"):].strip()
            
        # Extract final assistant response
        assistant_response = conversations[3]["value"] if len(conversations) > 3 else ""
        
        # Get image path from data
        image_path = data.get("image", "")
        
        # Determine modality from tool or data
        modality = None
        if tool_name in self.tool_modalities:
            modality = random.choice(self.tool_modalities[tool_name])
        
        return ToolExample(
            tool_name=tool_name,
            image_path=image_path,
            input_prompt=user_prompt,
            tool_params=tool_params,
            tool_output=tool_output,
            assistant_response=assistant_response,
            thoughts=thoughts,
            modality=modality
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
# Diversity planners -----------------------------------------------------------
# -----------------------------------------------------------------------------

class DiversityPlanner:
    """Plans conversation chains with diversity strategy."""
    
    def __init__(self, registry: ToolRegistry, bank: EnhancedSingleRoundBank):
        self.registry = registry
        self.bank = bank
        
        # Image introduction scenarios
        self.image_scenarios = {
            "same_image": 0.5,           # Continue with same image(s)
            "add_for_comparison": 0.2,   # Add second image for comparison
            "add_for_registration": 0.15, # Add second image for registration
            "new_image": 0.15            # Introduce completely new image
        }
        
        # Chain templates by length
        self.chain_templates = {
            "short": [
                ["UltraSAM", "LLaVA-Rad"],
                ["UniGradICON", "UltraSAM"], 
                ["LLaVA", "RaTE-NER"],
                ["HealthGPT", "SpecialistVLMs"],
                ["IterNet", "LLaVA-Rad"],
                ["LLaVA-Rad", "PMC-LLaMA"],
                ["SpecialistVLMs", "LLaVA"],
                ["BiomedCLIP", "LLaVA-Med"],              # Classification -> VQA
                ["Grounding-DINO", "MedSAM"],             # Grounding -> Segmentation
                ["MedSAM", "ChatCAD"],                    # Segmentation -> Report
                ["LLaVA-Med", "ChatCAD"],                 # VQA -> Report
                ["G-Seg", "LLaVA-Med"],                   # Combined Grounding+Seg -> VQA
                ["ChatCAD", "ChatCAD+"],                  # Report -> RAG
                ["BiomedCLIP", "ChatCAD"],                # Classification -> Report
                ["LLaVA-Med", "PMC-LLaMA"],               # VQA -> Medical QA
            ],
            "medium": [
                ["UniGradICON", "UltraSAM", "LLaVA-Rad"],
                ["HealthGPT", "LLaVA-Rad", "LLaVA", "RaTE-NER"],
                ["IterNet", "SpecialistVLMs", "PMC-LLaMA"],
                ["UltraSAM", "LLaVA-Rad", "PMC-LLaMA", "LLaVA"],
                ["UniGradICON", "LLaVA-Rad", "LLaVA", "PMC-LLaMA"],
                ["HealthGPT", "UltraSAM", "SpecialistVLMs"],
                ["BiomedCLIP", "Grounding-DINO", "MedSAM", "ChatCAD"],      # Classify -> Ground -> Segment -> Report
                ["LLaVA-Med", "G-Seg", "ChatCAD"],                         # VQA -> Combined Seg -> Report
                ["Grounding-DINO", "MedSAM", "LLaVA-Med", "ChatCAD+"],     # Ground -> Seg -> VQA -> RAG
                ["BiomedCLIP", "LLaVA-Med", "ChatCAD", "PMC-LLaMA"],       # Classify -> VQA -> Report -> QA
                ["HealthGPT", "BiomedCLIP", "LLaVA-Med"],                  # Enhance -> Classify -> VQA
                ["UltraSAM", "LLaVA-Med", "ChatCAD"],                      # Segment -> VQA -> Report
                ["UniGradICON", "BiomedCLIP", "ChatCAD"],                  # Register -> Classify -> Report
            ],
            "long": [
                ["UniGradICON", "UltraSAM", "LLaVA-Rad", "LLaVA", "RaTE-NER", "PMC-LLaMA"],
                ["HealthGPT", "UltraSAM", "LLaVA-Rad", "LLaVA", "RaTE-NER", "PMC-LLaMA"],
                ["IterNet", "SpecialistVLMs", "LLaVA", "RaTE-NER", "PMC-LLaMA"],
                ["UniGradICON", "LLaVA-Rad", "LLaVA", "RaTE-NER", "PMC-LLaMA", "SpecialistVLMs"],
                ["BiomedCLIP", "Grounding-DINO", "MedSAM", "LLaVA-Med", "ChatCAD", "ChatCAD+"],  # Full pipeline
                ["UniGradICON", "BiomedCLIP", "G-Seg", "LLaVA-Med", "ChatCAD", "PMC-LLaMA"],   # Registration workflow
                ["HealthGPT", "Grounding-DINO", "MedSAM", "LLaVA-Med", "ChatCAD", "PMC-LLaMA"], # Enhancement workflow
                ["BiomedCLIP", "LLaVA-Med", "G-Seg", "ChatCAD", "ChatCAD+", "PMC-LLaMA"],      # Analysis workflow
                ["UltraSAM", "BiomedCLIP", "LLaVA-Med", "ChatCAD", "RaTE-NER", "PMC-LLaMA"],  # Mixed workflow
            ]
        }
        
        # Diversity weights
        self.length_weights = {"short": 0.4, "medium": 0.4, "long": 0.2}
        
    def plan_conversation(self) -> List[str]:
        """Plan a diverse conversation chain."""
        # Select chain length
        length = random.choices(
            list(self.length_weights.keys()),
            weights=list(self.length_weights.values())
        )[0]
        
        # Select template from that length category
        templates = self.chain_templates[length]
        base_chain = random.choice(templates)
        
        # Verify tools have available data
        available_chain = []
        for tool in base_chain:
            example = self.bank.get_example(tool)
            if example:  # Only include tools with available data
                available_chain.append(tool)
            
        # Ensure minimum chain length
        if len(available_chain) < 2:
            # Fall back to any available tools
            all_tools = list(self.registry.tools.keys())
            available_tools = [t for t in all_tools if self.bank.get_example(t)]
            if len(available_tools) >= 2:
                available_chain = random.sample(available_tools, 2)
                
        return available_chain


# -----------------------------------------------------------------------------
# Context-aware conversation building ------------------------------------------
# -----------------------------------------------------------------------------

class ContextAwareBuilder:
    """Builds natural conversation with context awareness."""
    
    def __init__(self, registry: ToolRegistry, bank: EnhancedSingleRoundBank):
        self.registry = registry
        self.bank = bank

        # Scenario-specific tool preferences
        self.scenario_tools = {
            "registration": ["UniGradICON", "BiomedCLIP"],  # Tools that benefit from two images
            "comparison": ["LLaVA-Med", "BiomedCLIP", "ChatCAD"],  # Tools for comparative analysis
            "single_analysis": ["MedSAM", "G-Seg", "Grounding-DINO"],  # Deep single image analysis
            "text_based": ["ChatCAD+", "PMC-LLaMA", "RaTE-NER"],  # Can work without images
        }
        
    def build_conversation(self, planned_chain: List[str]) -> Dict[str, Any]:
        """Build a multi-round conversation from planned chain."""
        if not planned_chain:
            return {}
            
        # Initialize conversation state
        state = ConvState(session_id=str(uuid.uuid4()))
        conversations = []
        
        # Get base image from first tool
        first_example = self.bank.get_example(planned_chain[0])
        if not first_example:
            return {}
            
        # Initialize with first image
        if first_example.image_path:
            state.all_image_paths.append(first_example.image_path)
        state.current_modality = first_example.modality
        
        # Build each conversation turn
        for i, tool_name in enumerate(planned_chain):
            turn = self._build_turn(tool_name, state, i, planned_chain)
            if turn:
                conversations.extend(turn)
                state.tool_history.append(tool_name)
        
        # Prepare final output following Qwen format
        if len(state.all_image_paths) == 0:
            # Text-only conversation
            image_field = ""
        elif len(state.all_image_paths) == 1:
            # Single image
            image_field = state.all_image_paths[0]
        else:
            # Multiple images - use list format
            image_field = state.all_image_paths
        
        return {
            "image": image_field,
            "conversations": conversations
        }
    
    def _build_turn(self, tool_name: str, state: ConvState, turn_idx: int, chain: List[str]) -> List[Dict[str, Any]]:
        """Build a single conversation turn (user request + assistant response + tool output + final response)."""
        example = self.bank.get_example(tool_name)
        if not example:
            return []
            
        # Adapt user prompt for context and scenarios
        user_prompt = self._adapt_user_prompt_enhanced(example, state, turn_idx, chain, tool_name)
        
        # Create assistant tool call
        assistant_call = self._create_assistant_call_enhanced(example, state)
        
        # Create tool output
        tool_output = self._create_tool_output(example, state, tool_name)
        
        # Create final response
        final_response = self._create_final_response_enhanced(example, state)
        
        # Update state with artifacts
        self._update_state_with_artifacts_enhanced(state, tool_name, example)
        
        return [
            {"from": "human", "value": user_prompt},
            assistant_call,
            {"from": "human", "value": tool_output},
            final_response
        ]
    
    def _adapt_user_prompt_enhanced(self, example: ToolExample, state: ConvState, turn_idx: int, chain: List[str], tool_name: str) -> str:
        """Enhanced user prompt adaptation supporting all 4 scenarios."""
        
        if turn_idx == 0:
            # **Scenario 1 & 2**: First turn handling 
            if tool_name in ["UniGradICON", "BiomedCLIP"] and random.random() < 0.3:
                # **Scenario 2**: Start with two images for registration/comparison
                if example.image_path:
                    second_image_path = f"{example.image_path}_comparison"
                    state.all_image_paths.append(second_image_path)
                    state.has_second_image = True
                    comparison_prompts = [
                        f"<image>\n<image>\nCompare these two medical images and {example.input_prompt.lower()}",
                        f"<image>\n<image>\nI have two images for analysis. {example.input_prompt}",
                        f"<image>\n<image>\nPlease analyze both images: {example.input_prompt}",
                    ]
                    return random.choice(comparison_prompts)
            elif tool_name == "ChatCAD+" and random.random() < 0.4:
                # RAG can work without images
                return f"Based on medical literature, {example.input_prompt.lower()}"
            else:
                # **Scenario 1**: Normal single image start
                if example.image_path:
                    return f"<image>\n{example.input_prompt}"
                else:
                    return example.input_prompt
        else:
            # Later turns - implement scenarios 3 & 4
            scenario_choice = random.choices(
                ["same_image", "add_for_comparison", "add_for_registration", "new_image"],
                weights=[0.5, 0.2, 0.15, 0.15]
            )[0]
            
            if scenario_choice == "add_for_comparison" and not state.has_second_image and example.image_path:
                # **Scenario 3**: Add second image for comparison
                state.has_second_image = True
                second_image_path = f"{example.image_path}_comparison"
                state.all_image_paths.append(second_image_path)
                
                comparison_contexts = [
                    f"Now I have a second image to compare. <image>\n{example.input_prompt}",
                    f"Let me add another image for comparison. <image>\n{example.input_prompt}",
                    f"Here's a comparison image. <image>\nCan you {example.input_prompt.lower()}",
                    f"I want to compare this new image <image> with the previous one. {example.input_prompt}",
                ]
                return random.choice(comparison_contexts)
                
            elif scenario_choice == "add_for_registration" and tool_name == "UniGradICON" and not state.has_second_image and example.image_path:
                # **Scenario 3**: Add second image specifically for registration
                state.has_second_image = True 
                second_image_path = f"{example.image_path}_registration"
                state.all_image_paths.append(second_image_path)
                
                registration_contexts = [
                    f"Now I need to register this image <image> with the previous one. {example.input_prompt}",
                    f"Here's an additional image for registration. <image>\n{example.input_prompt}",
                    f"I want to align this new image <image> with the existing analysis. {example.input_prompt}",
                ]
                return random.choice(registration_contexts)
                
            elif scenario_choice == "new_image" and turn_idx >= 2 and example.image_path:
                # **Scenario 4**: Switch to completely different image
                new_image_path = example.image_path
                if new_image_path not in state.all_image_paths:
                    state.all_image_paths.append(new_image_path)
                    # Update modality if different
                    if example.modality and example.modality != state.current_modality:
                        state.current_modality = example.modality
                        modality_contexts = [
                            f"Now let me switch to a different {example.modality} image. <image>\n{example.input_prompt}",
                            f"I have a {example.modality} scan to analyze. <image>\n{example.input_prompt}",
                            f"Let's look at this {example.modality} image instead. <image>\n{example.input_prompt}",
                        ]
                        return random.choice(modality_contexts)
                    else:
                        switch_contexts = [
                            f"Now I have a different image to analyze. <image>\n{example.input_prompt}",
                            f"Let me switch to this new image. <image>\n{example.input_prompt}",
                            f"Here's another image I'd like you to examine. <image>\n{example.input_prompt}",
                        ]
                        return random.choice(switch_contexts)
            
            # **Scenario 1**: Default - continue with same image(s)
            continuation_contexts = [
                f"Now, {example.input_prompt.lower()}",
                f"Following up on the previous analysis, {example.input_prompt.lower()}",
                f"Next, {example.input_prompt.lower()}",
                f"Building on the results so far, {example.input_prompt.lower()}",
                f"Using the previous output, {example.input_prompt.lower()}",
                f"Based on what we found, {example.input_prompt.lower()}",
            ]
            return random.choice(continuation_contexts)
    
    def _create_assistant_call_enhanced(self, example: ToolExample, state: ConvState) -> Dict[str, Any]:
        """Enhanced assistant tool call with better context awareness."""
        # Enhanced thoughts based on conversation history and modality
        thoughts = example.thoughts
        
        if state.tool_history:
            tools_used = ", ".join(state.tool_history[-3:])  # Last 3 tools
            thoughts = f"Building on the previous {tools_used} analysis. {thoughts}"
            
        if state.current_modality:
            thoughts = f"Working with {state.current_modality} imaging. {thoughts}"
            
        if len(state.all_image_paths) > 1:
            thoughts = f"Analyzing multiple images ({len(state.all_image_paths)} total). {thoughts}"
            
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
    
    def _create_final_response_enhanced(self, example: ToolExample, state: ConvState) -> Dict[str, Any]:
        """Enhanced final assistant response with better context integration."""
        thoughts = f"Based on the {example.tool_name} output, I can now provide a comprehensive answer."
        
        if state.conversation_context:
            recent_context = "; ".join(state.conversation_context[-2:])
            thoughts += f" This builds on our previous analysis: {recent_context}"
            
        if len(state.all_image_paths) > 1:
            thoughts += f" I've analyzed {len(state.all_image_paths)} images in this conversation."
            
        return {
            "from": "gpt", 
            "thoughts": thoughts,
            "actions": [],
            "value": example.assistant_response
        }
    
    def _update_state_with_artifacts_enhanced(self, state: ConvState, tool_name: str, example: ToolExample) -> None:
        artifact_id = f"{tool_name.lower().replace('-', '_')}_{len(state.artifacts):03d}"
        
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
            "conch": "tissue_classification",
            "dsmil": "tumor_detection",
            "Cellvit": "cell_segmentation",
            "cellsam": "wsi_segmentation",
            "LLaVA-Med": "medical_vqa_response",      # Medical VQA response
            "BiomedCLIP": "medical_classification",   # Medical image classification
            "Grounding-DINO": "object_grounding",     # Grounded object detection
            "MedSAM": "medical_segmentation",         # Medical segmentation mask
            "G-Seg": "grounded_segmentation",         # Combined grounding + segmentation
            "ChatCAD": "medical_report",              # Medical report
            "ChatCAD+": "rag_medical_response",       # RAG-enhanced medical response
        }
        
        artifact_type = artifact_types.get(tool_name, "output")
        
        # Enhanced metadata
        metadata = {
            "params": example.tool_params,
            "modality": example.modality,
            "image_count": len(state.all_image_paths),
            "turn_index": len(state.tool_history)
        }
        
        artifact = Artifact(
            id=artifact_id,
            type=artifact_type,
            source_tool=tool_name,
            content=example.tool_output,
            metadata=metadata
        )
        
        state.add_artifact(artifact)
        
        # Enhanced context with modality info
        context_desc = f"{tool_name} generated {artifact_type}"
        if example.modality:
            context_desc += f" for {example.modality} imaging"
            
        state.add_context(context_desc)


# -----------------------------------------------------------------------------
# Main builder function -------------------------------------------------------
# -----------------------------------------------------------------------------

def build_enhanced_conversation(
    registry: ToolRegistry,
    bank: EnhancedSingleRoundBank
) -> Dict[str, Any]:
    """Build a single enhanced conversation with real data and diversity."""
    planner = DiversityPlanner(registry, bank)
    builder = ContextAwareBuilder(registry, bank)
    
    # Plan conversation
    chain = planner.plan_conversation()
    if not chain:
        return {}
        
    # Build conversation
    conversation = builder.build_conversation(chain)
    return conversation


# -----------------------------------------------------------------------------
# CLI interface ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multi-round Qwen-style dialogue datasets")
    parser.add_argument("--tool_yaml", type=str, required=True, help="Path to tool metadata YAML")
    parser.add_argument("--single_round_dir", type=str, required=True, help="Path to single-round examples directory")
    parser.add_argument("--out", type=str, required=True, help="Output file path")
    parser.add_argument("--num", type=int, required=True, help="Number of conversations to generate")
    args = parser.parse_args()

    registry = ToolRegistry(args.tool_yaml)
    bank = EnhancedSingleRoundBank(args.single_round_dir)

    conversations = []
    successful = 0
    
    for i in range(args.num):
        if i % 1000 == 0:
            print(f"Generated {i}/{args.num} conversations...")
            
        conversation = build_enhanced_conversation(registry, bank)
        if conversation:
            conversations.append(conversation)
            successful += 1

    with open(args.out, "w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + "\n")

    print(f"Generated {successful} successful conversations out of {args.num} attempts")
    print(f"Success rate: {successful/args.num*100:.1f}%")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
