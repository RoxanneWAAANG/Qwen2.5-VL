"""
python3 multi_tool_multiround.py \
--tool_yaml corpus_pack/tool_meta.yaml \
--single_round_dir tool_instruct \
--out multi_round/multi_tool_multiround.jsonl \
--num 10

{
  "session_id": …,
  "image_id": …,
  "image": …,
  "file_name": …,
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
        """Build logical tool succession mapping based on modality and task flow."""
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
            "SpecialistVLMs": ["LLaVA", "PMC-LLaMA"]  # Specialist → Summary/QA
        }
        
        # Add workflow successors
        for tool, successors in workflow_chains.items():
            mapping[tool].extend(successors)
        
        # Add some random cross-modal possibilities
        names = list(self.tools)
        for tool in names:
            # Add 2-3 random other tools as potential successors
            others = [t for t in names if t != tool and t not in mapping[tool]]
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
    base_image_id: Optional[str] = None
    base_image_path: Optional[str] = None
    all_image_paths: List[str] = field(default_factory=list)  # NEW: track all image paths
    has_second_image: bool = False  # NEW: track if we have a second image for registration
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
            "SpecialistVLMs": "svlms_fundus_dataset.jsonl"
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
        try:
            conversations = data.get("conversations", [])
            if len(conversations) < 3:  # Need at least user → assistant → user → assistant
                return None
                
            # Extract user prompt (first human message)
            user_prompt = conversations[0]["value"].replace("<image>\n", "").strip()
            
            # Extract assistant tool call
            assistant_call = conversations[1]
            thoughts = assistant_call.get("thoughts", "")
            actions = assistant_call.get("actions", [])
            tool_params = actions[0]["API_params"] if actions else {}
            
            # Extract tool output (second human message contains tool output)
            tool_output_msg = conversations[2]["value"]
            tool_output = tool_output_msg.split("Answer my first request:")[0].strip()
            if tool_output.startswith(f"{tool_name} output:"):
                tool_output = tool_output[len(f"{tool_name} output:"):].strip()
                
            # Extract final assistant response
            assistant_response = conversations[3]["value"] if len(conversations) > 3 else ""
            
            return ToolExample(
                tool_name=tool_name,
                image_id=data["image_id"],
                image_path=data["file_name"],
                input_prompt=user_prompt,
                tool_params=tool_params,
                tool_output=tool_output,
                assistant_response=assistant_response,
                thoughts=thoughts
            )
        except Exception as e:
            print(f"Error parsing example: {e}")
            return None
    
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
        
        # NEW: scenarios for image introduction
        self.image_scenarios = {
            "same_image": 0.6,      # Continue with same image(s)
            "add_for_registration": 0.2,  # Add second image for registration
            "new_image": 0.2        # Introduce completely new image
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
                ["SpecialistVLMs", "LLaVA"]
            ],
            "medium": [
                ["UniGradICON", "UltraSAM", "LLaVA-Rad"],
                ["HealthGPT", "LLaVA-Rad", "LLaVA", "RaTE-NER"],
                ["IterNet", "SpecialistVLMs", "PMC-LLaMA"],
                ["UltraSAM", "LLaVA-Rad", "PMC-LLaMA", "LLaVA"],
                ["UniGradICON", "LLaVA-Rad", "LLaVA", "PMC-LLaMA"],
                ["HealthGPT", "UltraSAM", "SpecialistVLMs"]
            ],
            "long": [
                ["UniGradICON", "UltraSAM", "LLaVA-Rad", "LLaVA", "RaTE-NER", "PMC-LLaMA"],
                ["HealthGPT", "UltraSAM", "LLaVA-Rad", "LLaVA", "RaTE-NER", "PMC-LLaMA"],
                ["IterNet", "SpecialistVLMs", "LLaVA", "RaTE-NER", "PMC-LLaMA"],
                ["UniGradICON", "LLaVA-Rad", "LLaVA", "RaTE-NER", "PMC-LLaMA", "SpecialistVLMs"]
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
            
        state.base_image_id = first_example.image_id
        state.base_image_path = first_example.image_path
        state.all_image_paths.append(first_example.image_path)  # NEW: add to list
        
        # Build each conversation turn
        for i, tool_name in enumerate(planned_chain):
            turn = self._build_turn(tool_name, state, i, planned_chain)
            if turn:
                conversations.extend(turn)
                state.tool_history.append(tool_name)
        
        # Prepare final output with proper image format
        image_field = state.all_image_paths if len(state.all_image_paths) > 1 else state.base_image_path
        
        return {
            "session_id": state.session_id,
            "image_id": state.base_image_id,
            "image": image_field,  # List if multiple images, string if single
            "file_name": state.base_image_path,  # Always the primary image
            "conversations": conversations
        }
    
    def _build_turn(self, tool_name: str, state: ConvState, turn_idx: int, chain: List[str]) -> List[Dict[str, Any]]:
        """Build a single conversation turn (user request + assistant response + tool output + final response)."""
        example = self.bank.get_example(tool_name)
        if not example:
            return []
            
        # Adapt user prompt for context
        user_prompt = self._adapt_user_prompt(example, state, turn_idx, chain, tool_name)
        
        # Create assistant tool call
        assistant_call = self._create_assistant_call(example, state)
        
        # Create tool output
        tool_output = self._create_tool_output(example, state, tool_name)
        
        # Create final response
        final_response = self._create_final_response(example, state)
        
        # Update state with artifacts
        self._update_state_with_artifacts(state, tool_name, example)
        
        return [
            {"from": "human", "value": user_prompt},
            assistant_call,
            {"from": "human", "value": tool_output},
            final_response
        ]
    
    def _adapt_user_prompt(self, example: ToolExample, state: ConvState, turn_idx: int, chain: List[str], tool_name: str) -> str:
        """Adapt user prompt for conversation context."""
        if turn_idx == 0:
            # First turn - handle registration special case
            if tool_name == "UniGradICON":
                # Registration needs two images
                second_image_path = f"{state.base_image_path}_ref"  # Simulate second image
                state.all_image_paths.append(second_image_path)
                state.has_second_image = True
                return f"<image>\n<image>\n{example.input_prompt}"
            else:
                # Normal single image
                if state.base_image_path:
                    return f"<image>\n{example.input_prompt}"
                else:
                    return example.input_prompt
        else:
            # Later turns - decide scenario
            if tool_name == "UniGradICON" and not state.has_second_image:
                # Registration in later turn - introduce second image
                state.has_second_image = True
                # Add second image path (simulate or use from example)
                second_image_path = f"{state.base_image_path}_ref"  # Simulate second image
                state.all_image_paths.append(second_image_path)
                context_phrases = [
                    f"Now I have a second image. <image>\n{example.input_prompt}",
                    f"Here's an additional image for registration. <image>\n{example.input_prompt}",
                    f"I want to register this new image <image> with the previous one. {example.input_prompt}"
                ]
                return random.choice(context_phrases)
            elif turn_idx >= 2 and random.random() < 0.3:  # NEW: 30% chance to introduce new image after turn 2
                # Introduce completely new image
                new_image_path = example.image_path  # Use example's image as new image
                if new_image_path not in state.all_image_paths:
                    state.all_image_paths.append(new_image_path)
                    context_phrases = [
                        f"Now I have a different image to analyze. <image>\n{example.input_prompt}",
                        f"Let me switch to this new image. <image>\n{example.input_prompt}",
                        f"Here's another image I'd like you to examine. <image>\n{example.input_prompt}",
                        f"I want to analyze this different image now. <image>\n{example.input_prompt}"
                    ]
                    return random.choice(context_phrases)
                else:
                    # Fall back to normal context if image already exists
                    pass
            
            # Normal follow-up (same image continuation)
            context_phrases = [
                f"Now, {example.input_prompt.lower()}",
                f"Following up on the previous analysis, {example.input_prompt.lower()}",
                f"Next, {example.input_prompt.lower()}",
                f"Building on the results so far, {example.input_prompt.lower()}",
                f"Using the previous output, {example.input_prompt.lower()}"
            ]
            return random.choice(context_phrases)
    
    def _create_assistant_call(self, example: ToolExample, state: ConvState) -> Dict[str, Any]:
        """Create assistant tool call response."""
        # Adapt thoughts for context
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
            thoughts += f" This builds on our previous analysis: {'; '.join(state.conversation_context[-2:])}"
            
        return {
            "from": "gpt", 
            "thoughts": thoughts,
            "actions": [],
            "value": example.assistant_response
        }
    
    def _update_state_with_artifacts(self, state: ConvState, tool_name: str, example: ToolExample) -> None:
        """Update conversation state with new artifacts."""
        # Create artifact based on tool type
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
            "SpecialistVLMs": "specialist_report"
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
    for _ in range(args.num):
        conversation = build_enhanced_conversation(registry, bank)
        if conversation:
            conversations.append(conversation)

    with open(args.out, "w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + "\n")

    print(f"Generated {len(conversations)} conversations and saved to {args.out}")


if __name__ == "__main__":
    main()
