#!/usr/bin/env python3
"""
Generate multi-tool single-round conversations
Format: User request → Assistant uses multiple tools → Tool outputs → Final response

python3 multi_tool_single_round.py \
--tool_yaml corpus_pack/tool_meta.yaml \
--single_round_dir tool_instruct \
--out multi_round/multi_tool_single_round.jsonl \
--num 20000
"""

import argparse
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# -----------------------------------------------------------------------------
# Tool metadata and data extraction (reuse from previous script)
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
    def __init__(self, yaml_path: Union[str, Path]):
        with Path(yaml_path).open() as f:
            raw = yaml.safe_load(f)
        self.tools: Dict[str, Tool] = {name: Tool.from_dict(name, cfg) for name, cfg in raw.items()}

    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]

@dataclass
class ToolExample:
    tool_name: str
    image_id: str
    image_path: str
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
            "conch": "instruction_2000_img_updated_conch.jsonl",
            "dsmil": "instruction_2000_img_updated_conch.jsonl",
            "Cellvit": "instruction_2000_img_updated_conch.jsonl",
            "cellsam": "instruction_2000_img_updated_conch.jsonl"
        }
        self._cache: Dict[str, List[ToolExample]] = {}

    def load_tool_examples(self, tool_name: str, max_examples: int = 100) -> List[ToolExample]:
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
        conversations = data.get("conversations", [])
        if len(conversations) < 3:
            return None

        # Check if this example is for the requested tool
        assistant_call = conversations[1]
        actions = assistant_call.get("actions", [])
        if not actions:
            return None

        actual_tool_name = actions[0].get("API_name", "")
        if actual_tool_name != tool_name:
            return None

        user_prompt = conversations[0]["value"].replace("<image>\n", "").strip()
        thoughts = assistant_call.get("thoughts", "")
        tool_params = actions[0]["API_params"] if actions else {}

        tool_output_msg = conversations[2]["value"]
        tool_output = tool_output_msg.split("Answer my first request:")[0].strip()
        if tool_output.startswith(f"{tool_name} output:"):
            tool_output = tool_output[len(f"{tool_name} output:"):].strip()

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

    def get_random_example(self, tool_name: str) -> Optional[ToolExample]:
        examples = self.load_tool_examples(tool_name)
        return random.choice(examples) if examples else None

# -----------------------------------------------------------------------------
# Multi-tool combination logic
# -----------------------------------------------------------------------------

class MultiToolPlanner:
    """Plans logical multi-tool combinations within a single round."""
    
    def __init__(self, registry: ToolRegistry, extractor: RealDataExtractor):
        self.registry = registry
        self.extractor = extractor
        
        # Define logical tool combinations
        self.tool_combinations = {
            # Segmentation → Analysis workflows
            "segment_and_analyze": [
                ["UltraSAM", "LLaVA-Rad"],  # Segment → Generate report
                ["IterNet", "SpecialistVLMs"],  # Fundus segment → Specialist analysis
                ["Cellvit", "conch"],  # Cell segment → Tissue classification  
                ["cellsam", "dsmil"],  # WSI segment → Tumor detection
                ["UltraSAM", "LLaVA"],  # Segment → Summarize
            ],
            
            # Detection → Classification workflows  
            "detect_and_classify": [
                ["dsmil", "conch"],  # Detect tumor → Classify tissue
                ["conch", "Cellvit"],  # Classify tissue → Segment cells
                ["dsmil", "LLaVA-Rad"],  # Detect tumor → Generate report
            ],
            
            # Enhancement → Analysis workflows
            "enhance_and_analyze": [
                ["HealthGPT", "LLaVA-Rad"],  # Enhance → Analyze
                ["HealthGPT", "UltraSAM"],  # Enhance → Segment
                ["UniGradICON", "LLaVA-Rad"],  # Register → Analyze
            ],
            
            # Analysis → Summarization workflows
            "analyze_and_summarize": [
                ["LLaVA-Rad", "LLaVA"],  # Generate report → Summarize
                ["SpecialistVLMs", "LLaVA"],  # Specialist analysis → Summarize
                ["LLaVA-Rad", "RaTE-NER"],  # Generate report → Extract entities
                ["LLaVA", "PMC-LLaMA"],  # Summarize → Medical QA
            ],
            
            # Complex 3-tool workflows
            "complex_workflow": [
                ["UltraSAM", "LLaVA-Rad", "LLaVA"],  # Segment → Report → Summarize
                ["dsmil", "conch", "LLaVA-Rad"],  # Detect → Classify → Report
                ["HealthGPT", "UltraSAM", "LLaVA"],  # Enhance → Segment → Summarize
                ["Cellvit", "dsmil", "conch"],  # Segment cells → Detect tumor → Classify
            ]
        }
        
        # Combination weights
        self.combination_weights = {
            "segment_and_analyze": 0.3,
            "detect_and_classify": 0.25,
            "enhance_and_analyze": 0.2,
            "analyze_and_summarize": 0.15,
            "complex_workflow": 0.1
        }
        
    def plan_multi_tool_combination(self) -> List[str]:
        """Plan a multi-tool combination."""
        # Select combination type
        combo_type = random.choices(
            list(self.combination_weights.keys()),
            weights=list(self.combination_weights.values())
        )[0]
        
        # Select specific combination
        combinations = self.tool_combinations[combo_type]
        selected_combo = random.choice(combinations)
        
        # Verify all tools have available data
        available_tools = []
        for tool in selected_combo:
            if self.extractor.get_random_example(tool):
                available_tools.append(tool)
                
        # Ensure we have at least 2 tools
        if len(available_tools) < 2:
            # Fallback to any available tools
            all_tools = list(self.registry.tools.keys())
            available_all = [t for t in all_tools if self.extractor.get_random_example(t)]
            if len(available_all) >= 2:
                available_tools = random.sample(available_all, 2)
                
        return available_tools

# -----------------------------------------------------------------------------
# Multi-tool conversation builder
# -----------------------------------------------------------------------------

class MultiToolConversationBuilder:
    """Builds conversations with multiple tools in a single round."""
    
    def __init__(self, registry: ToolRegistry, extractor: RealDataExtractor):
        self.registry = registry
        self.extractor = extractor
        
        # Highly diversified templates for complex questions that naturally require multiple tools
        self.request_templates = {
            "pathology_comprehensive": [
                # Clinical assessment variations
                "I need a thorough pathological assessment of this specimen. What do you see?",
                "Can you provide a comprehensive histopathological evaluation of this slide?",
                "What's your complete diagnostic impression of this tissue sample?",
                "I'm looking for a detailed pathology workup - what are your findings?",
                "Could you give me a full microscopic examination report for this case?",
                
                # Concern-based inquiries
                "I'm worried about malignancy in this biopsy. What's your assessment?",
                "There might be something concerning here - can you take a look?",
                "I need to rule out cancer in this specimen. What do you think?",
                "This patient has concerning symptoms - what does the histology show?",
                "I'm seeing some abnormal areas - can you help me evaluate them?",
                
                # Educational/learning context
                "This is an interesting case for our tumor board - what's your analysis?",
                "I'm presenting this at our pathology conference - help me understand the findings.",
                "This is a teaching case - can you walk through the diagnostic features?",
                "Our residents are struggling with this case - what would you highlight?",
                "I want to use this for medical education - what are the key findings?"
            ],
            
            "radiology_comprehensive": [
                # Diagnostic urgency variations
                "STAT read needed - what are the critical findings in this scan?",
                "Emergency case - I need your immediate assessment of this image.",
                "Patient is in the ER - what's your rapid interpretation?",
                "Urgent consultation needed - what do you see here?",
                "Time-sensitive case - please provide your diagnostic impression.",
                
                # Routine clinical scenarios  
                "Can you help me interpret this routine imaging study?",
                "I need a comprehensive read of this scan for my patient.",
                "What's your assessment of this imaging for our weekly review?",
                "Standard workup case - what are your findings?",
                "Regular follow-up scan - any changes or concerns?",
                
                # Comparison and follow-up
                "How does this compare to previous imaging? Any progression?",
                "I need to assess treatment response - what do you see?",
                "Post-operative follow-up - is there anything concerning?",
                "Surveillance imaging - any new developments?",
                "Pre-treatment planning - what's the current status?"
            ],
            
            "quality_enhancement": [
                # Image quality issues
                "This scan has some technical issues - can you still extract useful information?",
                "The image quality isn't ideal - what can you determine?",
                "Motion artifacts are present - help me see through the noise.",
                "Low-resolution study - can you enhance and analyze?",
                "Suboptimal imaging conditions - what diagnostic info can you get?",
                
                # Enhancement for better visualization
                "I need clearer visualization of the anatomical structures here.",
                "Can you optimize this image for better diagnostic clarity?",
                "The contrast isn't great - can you help me see the details better?",
                "Enhancement needed for accurate measurement and assessment.",
                "I need the best possible image quality for precise diagnosis."
            ],
            
            "multi_modal_analysis": [
                # Complex diagnostic scenarios
                "This is a challenging case requiring multiple analytical approaches.",
                "I need the most comprehensive evaluation possible for this complex patient.",
                "Multidisciplinary case - what can you contribute from imaging perspective?",
                "Difficult diagnosis - I need all available analytical tools applied.",
                "Complex presentation - help me piece together the findings.",
                
                # Research and academic context
                "Research case - I need detailed quantitative and qualitative analysis.",
                "Publication-quality assessment needed - what are your findings?",
                "Grant application case study - provide comprehensive analysis.",
                "Academic presentation - what would you emphasize?",
                "Scientific documentation needed - thorough evaluation required."
            ],
            
            "clinical_decision_support": [
                # Treatment planning
                "Treatment planning case - what information do you need to provide?",
                "Surgical planning - help me understand the anatomical relationships.",
                "Therapy selection depends on these findings - what do you see?",
                "Pre-procedural assessment - any contraindications or concerns?",
                "Management decisions hinge on this analysis - please be thorough.",
                
                # Patient communication prep
                "I need to explain these findings to the patient - help me understand them.",
                "Family consultation tomorrow - what should I highlight?",
                "Patient education case - what are the key points to communicate?",
                "Informed consent discussion - what risks should I mention?",
                "Prognosis discussion preparation - what does this tell us?"
            ],
            
            "second_opinion": [
                # Peer consultation
                "Second opinion needed - do you agree with my initial assessment?",
                "Colleague asked me to review this - what's your take?",
                "Quality assurance case - please provide independent analysis.",
                "I want to confirm my findings - what do you see?",
                "External review requested - can you provide your assessment?",
                
                # Challenging cases
                "I'm uncertain about this case - can you help clarify?",
                "Atypical presentation - what's your diagnostic impression?",
                "Borderline findings - how would you interpret this?",
                "Equivocal results - what additional insights can you provide?",
                "Diagnostic dilemma - what's your analysis?"
            ],
            
            "registration_workflow": [
                # Temporal comparison
                "I have these two studies from different time points - what's changed?",
                "Progression assessment needed - how do these scans compare?",
                "Before and after treatment - what differences do you see?",
                "Serial imaging evaluation - any concerning developments?",
                "Longitudinal analysis required - what's the evolution?",
                
                # Multi-sequence analysis
                "I need to correlate findings across these different sequences.",
                "Cross-sectional analysis of these related images needed.",
                "Comparative assessment of these complementary studies.",
                "Integration of findings from these multiple acquisitions.",
                "Synthesis needed across these different imaging approaches."
            ],
            
            "screening_detection": [
                # Preventive care
                "Routine screening case - any abnormalities detected?",
                "Population health screening - what's your assessment?",
                "Early detection protocol - are there any concerning findings?",
                "Asymptomatic patient screening - anything to flag?",
                "Preventive imaging evaluation - what do you recommend?",
                
                # High-risk patients
                "High-risk patient surveillance - any new developments?",
                "Genetic predisposition screening - what do you see?",
                "Occupational health screening - any exposure-related changes?",
                "Family history concern - is there anything suspicious?",
                "Risk stratification case - what's your assessment?"
            ],
            
            "research_academic": [
                # Scientific analysis
                "Research protocol case - I need quantitative measurements.",
                "Clinical trial imaging - what are the objective findings?",
                "Biomarker study - can you extract relevant features?",
                "Outcome prediction research - what prognostic indicators do you see?",
                "Machine learning validation - what ground truth can you provide?",
                
                # Educational content
                "Medical school teaching case - what would you emphasize?",
                "Residency training material - highlight the learning points.",
                "CME presentation case - what are the key takeaways?",
                "Board exam preparation - what should students focus on?",
                "Continuing education case - what's clinically relevant?"
            ],
            
            "subspecialty_focused": [
                # Specialized domains
                "Pediatric case - what age-specific considerations apply?",
                "Geriatric patient - any age-related findings?",
                "Oncology case - staging and prognostic information needed.",
                "Cardiovascular focus - what hemodynamic insights can you provide?",
                "Neurological assessment - any functional implications?",
                
                # Specialized techniques
                "Molecular pathology correlation needed - what do you see?",
                "Immunohistochemistry guidance - what would you recommend?",
                "Advanced imaging protocol - specialized analysis required.",
                "Functional imaging interpretation - what does this reveal?",
                "Interventional planning - what anatomical details are crucial?"
            ]
        }
        
        # Task descriptions for each tool
        self.task_descriptions = {
            "UltraSAM": "segment the ultrasound image",
            "IterNet": "perform fundus segmentation", 
            "LLaVA-Rad": "generate a radiological report",
            "LLaVA": "provide a summary",
            "SpecialistVLMs": "conduct specialist analysis",
            "HealthGPT": "enhance the image quality",
            "UniGradICON": "register the images",
            "RaTE-NER": "extract medical entities",
            "PMC-LLaMA": "answer medical questions",
            "conch": "classify the tissue type",
            "dsmil": "detect any tumors",
            "Cellvit": "segment the cells",
            "cellsam": "analyze the whole slide image"
        }
        
    def build_multi_tool_conversation(self, tool_chain: List[str]) -> Dict[str, Any]:
        """Build a conversation using multiple tools in one round."""
        if len(tool_chain) < 2:
            return {}
            
        # Get examples for all tools
        examples = []
        for tool in tool_chain:
            example = self.extractor.get_random_example(tool)
            if not example:
                return {}
            examples.append(example)
            
        # Use the first example for base metadata
        base_example = examples[0]
        
        # Create user request
        user_request = self._create_multi_tool_request(tool_chain, examples)
        
        # Create assistant response with multiple tools
        assistant_response = self._create_multi_tool_assistant_response(tool_chain, examples)
        
        # Create tool outputs  
        tool_output_message = self._create_multi_tool_output(tool_chain, examples)
        
        # Create final assistant response
        final_response = self._create_final_multi_tool_response(tool_chain, examples)
        
        return {
            "session_id": str(uuid.uuid4()),
            "image_id": base_example.image_id,
            "image": base_example.image_path,
            "file_name": base_example.image_path,
            "conversations": [
                {"from": "human", "value": user_request},
                assistant_response,
                {"from": "human", "value": tool_output_message}, 
                final_response
            ]
        }
        
    def _create_multi_tool_request(self, tool_chain: List[str], examples: List[ToolExample]) -> str:
        """Create a complex user question that naturally requires multiple tools."""
        # Handle special cases for image registration
        if "UniGradICON" in tool_chain:
            image_prefix = "<image>\n<image>\n"  # Two images for registration
            template_category = "registration_workflow"
        else:
            image_prefix = "<image>\n"
            
            # Determine template category based on tool combination with more diversity
            pathology_tools = {"Cellvit", "cellsam", "dsmil", "conch"}
            segmentation_tools = {"UltraSAM", "IterNet", "Cellvit"}
            analysis_tools = {"LLaVA-Rad", "SpecialistVLMs", "conch"}
            enhancement_tools = {"HealthGPT", "UniGradICON"}
            summarization_tools = {"LLaVA", "RaTE-NER", "PMC-LLaMA"}
            
            # Primary category selection with weighted randomness
            if any(tool in pathology_tools for tool in tool_chain):
                # Pathology cases have multiple subcategories
                pathology_categories = ["pathology_comprehensive", "subspecialty_focused", "second_opinion", "research_academic"]
                if any(tool in ["dsmil"] for tool in tool_chain):
                    pathology_categories.extend(["screening_detection", "clinical_decision_support"])
                template_category = random.choice(pathology_categories)
                
            elif any(tool in segmentation_tools for tool in tool_chain) and any(tool in analysis_tools for tool in tool_chain):
                # Segmentation + Analysis workflows
                seg_analysis_categories = ["radiology_comprehensive", "clinical_decision_support", "multi_modal_analysis"]
                template_category = random.choice(seg_analysis_categories)
                
            elif any(tool in enhancement_tools for tool in tool_chain):
                # Enhancement workflows  
                enhancement_categories = ["quality_enhancement", "clinical_decision_support"]
                template_category = random.choice(enhancement_categories)
                
            elif any(tool in analysis_tools for tool in tool_chain) and any(tool in summarization_tools for tool in tool_chain):
                # Analysis + Summarization workflows
                analysis_categories = ["radiology_comprehensive", "second_opinion", "clinical_decision_support"]
                template_category = random.choice(analysis_categories)
                
            elif len(tool_chain) >= 3:
                # Complex multi-tool workflows
                complex_categories = ["multi_modal_analysis", "research_academic", "subspecialty_focused"]
                template_category = random.choice(complex_categories)
                
            else:
                # Default categories with variety
                default_categories = ["radiology_comprehensive", "pathology_comprehensive", "clinical_decision_support"]
                template_category = random.choice(default_categories)
            
        # Select appropriate template
        if template_category in self.request_templates:
            request_text = random.choice(self.request_templates[template_category])
        else:
            # Fallback with variety
            fallback_options = [
                "Can you provide a comprehensive analysis of this medical image?",
                "I need a thorough assessment of this case.",
                "What's your diagnostic impression of this study?",
                "Help me understand what's happening in this image.",
                "I need your expert analysis of these findings."
            ]
            request_text = random.choice(fallback_options)
            
        return image_prefix + request_text
        
    def _create_multi_tool_assistant_response(self, tool_chain: List[str], examples: List[ToolExample]) -> Dict[str, Any]:
        """Create assistant response with multiple tool actions."""
        # Create diverse reasoning patterns
        reasoning_styles = [
            "systematic_clinical", "problem_solving", "educational", 
            "urgent_assessment", "comprehensive_analysis", "methodical_approach"
        ]
        style = random.choice(reasoning_styles)
        
        thoughts = self._generate_diverse_thoughts(tool_chain, style)
        
        # Create actions for all tools
        actions = []
        for i, (tool, example) in enumerate(zip(tool_chain, examples)):
            actions.append({
                "API_name": tool,
                "API_params": example.tool_params
            })
            
        # Create diverse value responses
        value = self._generate_diverse_value_response(tool_chain, style)
        
        return {
            "from": "gpt",
            "thoughts": thoughts,
            "actions": actions, 
            "value": value
        }
        
    def _generate_diverse_thoughts(self, tool_chain: List[str], style: str) -> str:
        """Generate diverse reasoning patterns based on style."""
        tool_count = len(tool_chain)
        
        if style == "systematic_clinical":
            thoughts = f"For a thorough clinical assessment, I'll employ a {tool_count}-step systematic approach. "
            if tool_count == 2:
                thoughts += f"I'll begin with {tool_chain[0]} to {self._get_tool_reasoning(tool_chain[0])}, "
                thoughts += f"followed by {tool_chain[1]} to {self._get_tool_reasoning(tool_chain[1])}. "
                thoughts += "This systematic workflow ensures comprehensive evaluation."
            else:
                thoughts += f"The sequence will be: {' → '.join(tool_chain)}, providing complete diagnostic information."
                
        elif style == "problem_solving":
            thoughts = f"This clinical question requires a multi-faceted analytical approach. "
            thoughts += f"I'll tackle this by using {tool_count} complementary tools. "
            thoughts += f"Starting with {tool_chain[0]} to establish baseline findings, "
            if tool_count > 1:
                thoughts += f"then proceeding with {tool_chain[1]} for additional insights. "
            thoughts += "This problem-solving strategy should provide the complete picture."
            
        elif style == "educational":
            thoughts = f"From a diagnostic perspective, this case requires {tool_count} analytical steps. "
            thoughts += f"First, {tool_chain[0]} will help us {self._get_tool_reasoning(tool_chain[0])}, "
            if tool_count > 1:
                thoughts += f"then {tool_chain[1]} will allow us to {self._get_tool_reasoning(tool_chain[1])}. "
            thoughts += "This educational approach demonstrates proper diagnostic methodology."
            
        elif style == "urgent_assessment":
            thoughts = f"Given the clinical urgency, I need to rapidly deploy {tool_count} tools for immediate assessment. "
            thoughts += f"Quick analysis with {tool_chain[0]} first, then rapid follow-up with {tool_chain[1] if tool_count > 1 else 'additional tools'}. "
            thoughts += "Time-efficient but thorough evaluation is essential."
            
        elif style == "comprehensive_analysis":
            thoughts = f"A comprehensive evaluation requires integrating multiple analytical approaches. "
            thoughts += f"I'll conduct a detailed {tool_count}-tool analysis: "
            thoughts += f"{', '.join([f'{tool} for {self._get_tool_reasoning(tool)}' for tool in tool_chain])}. "
            thoughts += "This comprehensive strategy ensures no diagnostic detail is missed."
            
        else:  # methodical_approach
            thoughts = f"I'll approach this methodically using {tool_count} specialized tools. "
            thoughts += f"Step 1: {tool_chain[0]} - {self._get_tool_reasoning(tool_chain[0])}. "
            if tool_count > 1:
                thoughts += f"Step 2: {tool_chain[1]} - {self._get_tool_reasoning(tool_chain[1])}. "
            if tool_count > 2:
                thoughts += f"Step 3: {tool_chain[2]} - {self._get_tool_reasoning(tool_chain[2])}. "
            thoughts += "This methodical approach ensures systematic evaluation."
            
        return thoughts
        
    def _generate_diverse_value_response(self, tool_chain: List[str], style: str) -> str:
        """Generate diverse value responses based on style."""
        tool_count = len(tool_chain)
        
        response_templates = {
            "systematic_clinical": [
                f"I'll conduct a systematic {tool_count}-step clinical analysis to provide you with comprehensive findings.",
                f"Let me perform a thorough clinical evaluation using {tool_count} specialized analytical tools.",
                f"I'll systematically analyze this case using multiple diagnostic approaches for complete assessment."
            ],
            "problem_solving": [
                f"This requires a strategic multi-tool approach - I'll solve this step by step using {tool_count} different analyses.",
                f"Let me break down this complex case using {tool_count} complementary analytical methods.",
                f"I'll tackle this diagnostic challenge using multiple specialized tools for comprehensive insights."
            ],
            "educational": [
                f"This is an excellent case for demonstrating multi-tool analysis - let me walk through the {tool_count}-step process.",
                f"I'll demonstrate proper diagnostic methodology using {tool_count} different analytical approaches.",
                f"This case showcases how multiple tools work together - let me show you the complete workflow."
            ],
            "urgent_assessment": [
                f"I'll provide rapid but thorough assessment using {tool_count} tools for immediate clinical insights.",
                f"Quick multi-tool analysis incoming - {tool_count} steps for fast but comprehensive evaluation.",
                f"Time-sensitive analysis using {tool_count} specialized tools for urgent diagnostic information."
            ],
            "comprehensive_analysis": [
                f"I'll provide the most comprehensive analysis possible using {tool_count} different specialized tools.",
                f"Complete diagnostic workup requires {tool_count} analytical approaches - let me process this thoroughly.",
                f"This case deserves comprehensive evaluation - I'll use {tool_count} tools for complete assessment."
            ],
            "methodical_approach": [
                f"I'll work through this methodically using {tool_count} analytical steps for precise evaluation.",
                f"Methodical analysis requires {tool_count} sequential steps - let me process this systematically.",
                f"I'll approach this case step-by-step using {tool_count} specialized tools for accurate diagnosis."
            ]
        }
        
        return random.choice(response_templates[style])
        
    def _get_tool_reasoning(self, tool_name: str) -> str:
        """Get reasoning for why each tool is needed."""
        reasoning_map = {
            "UltraSAM": "identify and segment the anatomical structures",
            "IterNet": "perform detailed fundus analysis",
            "LLaVA-Rad": "generate comprehensive radiological findings",
            "LLaVA": "provide clear summary and interpretation",
            "SpecialistVLMs": "conduct specialized medical analysis",
            "HealthGPT": "enhance image quality for better visualization",
            "UniGradICON": "align and register the images properly",
            "RaTE-NER": "extract specific medical entities and terminology",
            "PMC-LLaMA": "provide evidence-based medical insights",
            "conch": "determine tissue type and characteristics",
            "dsmil": "detect any tumor or abnormal regions",
            "Cellvit": "analyze cellular structures and boundaries",
            "cellsam": "examine the complete slide architecture"
        }
        return reasoning_map.get(tool_name, f"analyze using {tool_name}")
        
    def _create_multi_tool_output(self, tool_chain: List[str], examples: List[ToolExample]) -> str:
        """Create tool output message for multiple tools."""
        outputs = []
        for tool, example in zip(tool_chain, examples):
            outputs.append(f"{tool} output: {example.tool_output}")
            
        # Combine all outputs
        combined_output = "\n\n".join(outputs)
        
        # Reference the original complex question rather than explicit tool request
        original_question = self._get_original_question_context(tool_chain, examples)
        return f"{combined_output}\n\nAnswer my original question: {original_question}"
        
    def _get_original_question_context(self, tool_chain: List[str], examples: List[ToolExample]) -> str:
        """Generate a contextual reference to the original question with diversity."""
        # Different ways to reference the original question
        reference_styles = [
            "direct_question", "clinical_inquiry", "assessment_request", 
            "consultation_query", "diagnostic_question", "evaluation_request"
        ]
        style = random.choice(reference_styles)
        
        # Generate diverse question references based on tool combination
        pathology_tools = {"Cellvit", "cellsam", "dsmil", "conch"}
        radiology_tools = {"UltraSAM", "IterNet", "LLaVA-Rad", "SpecialistVLMs"}
        enhancement_tools = {"HealthGPT", "UniGradICON"}
        
        if any(tool in pathology_tools for tool in tool_chain):
            if style == "direct_question":
                questions = ["What do you see in this pathology slide?", "What's your assessment of this tissue sample?", "Can you evaluate this biopsy?"]
            elif style == "clinical_inquiry":
                questions = ["What are your findings on this specimen?", "What's your diagnostic impression?", "What pathological changes do you observe?"]
            elif style == "assessment_request":
                questions = ["Please assess this histological sample.", "I need your evaluation of this tissue.", "Can you analyze this pathology case?"]
            elif style == "consultation_query":
                questions = ["What's your opinion on this case?", "I'd like your consultation on these findings.", "Can you provide your expert assessment?"]
            elif style == "diagnostic_question":
                questions = ["What's your diagnosis based on these findings?", "What diagnostic conclusions can you draw?", "What's your interpretation of this case?"]
            else:  # evaluation_request
                questions = ["Please evaluate this specimen thoroughly.", "I need a comprehensive assessment.", "Can you provide detailed findings?"]
                
        elif any(tool in radiology_tools for tool in tool_chain):
            if style == "direct_question":
                questions = ["What can you tell me about this medical image?", "What do you see in this scan?", "What are the findings?"]
            elif style == "clinical_inquiry":
                questions = ["What's your radiological assessment?", "What are the imaging findings?", "What does this scan show?"]
            elif style == "assessment_request":
                questions = ["Please interpret this imaging study.", "I need your assessment of this scan.", "Can you evaluate this medical image?"]
            elif style == "consultation_query":
                questions = ["What's your reading of this study?", "I'd like your opinion on these images.", "Can you provide your interpretation?"]
            elif style == "diagnostic_question":
                questions = ["What's your diagnostic impression?", "What diagnosis do these findings suggest?", "What's your interpretation?"]
            else:  # evaluation_request
                questions = ["Please provide a comprehensive read.", "I need detailed imaging analysis.", "Can you give me a thorough assessment?"]
                
        elif "UniGradICON" in tool_chain:
            if style == "direct_question":
                questions = ["How do these images compare?", "What differences do you see?", "What's changed between these scans?"]
            elif style == "clinical_inquiry":
                questions = ["What's the progression between these studies?", "How do these time points compare?", "What evolution do you observe?"]
            elif style == "assessment_request":
                questions = ["Please compare these sequential images.", "I need comparison of these studies.", "Can you assess the changes?"]
            elif style == "consultation_query":
                questions = ["What's your opinion on the progression?", "How would you interpret these changes?", "What's your assessment of evolution?"]
            elif style == "diagnostic_question":
                questions = ["What do these changes suggest?", "What's the significance of these differences?", "What's your interpretation of progression?"]
            else:  # evaluation_request
                questions = ["Please evaluate the temporal changes.", "I need assessment of progression.", "Can you analyze the evolution?"]
                
        else:
            # General medical imaging questions
            if style == "direct_question":
                questions = ["What's your analysis of this case?", "What do you see here?", "What are your findings?"]
            elif style == "clinical_inquiry":
                questions = ["What's your clinical assessment?", "What are the significant findings?", "What's your medical opinion?"]
            elif style == "assessment_request":
                questions = ["Please assess this medical case.", "I need your evaluation.", "Can you analyze this study?"]
            elif style == "consultation_query":
                questions = ["What's your professional opinion?", "I'd like your consultation.", "Can you provide your expertise?"]
            elif style == "diagnostic_question":
                questions = ["What's your diagnostic thinking?", "What diagnosis do you suggest?", "What's your clinical impression?"]
            else:  # evaluation_request
                questions = ["Please provide comprehensive analysis.", "I need thorough evaluation.", "Can you give detailed assessment?"]
        
        return random.choice(questions)
        
    def _create_final_multi_tool_response(self, tool_chain: List[str], examples: List[ToolExample]) -> Dict[str, Any]:
        """Create final assistant response combining all tool results.""" 
        # Diverse response styles
        response_styles = ["clinical_report", "integrated_summary", "diagnostic_synthesis", 
                         "educational_explanation", "structured_findings", "narrative_assessment"]
        style = random.choice(response_styles)
        
        # Create thoughts showing integration of results
        thoughts = self._generate_integration_thoughts(tool_chain, style)
        
        # Create diverse response formats
        combined_response = self._generate_integrated_response(tool_chain, examples, style)
        
        return {
            "from": "gpt",
            "thoughts": thoughts,
            "actions": [],
            "value": combined_response
        }
        
    def _generate_integration_thoughts(self, tool_chain: List[str], style: str) -> str:
        """Generate diverse integration thoughts based on style."""
        tool_count = len(tool_chain)
        
        thoughts_templates = {
            "clinical_report": f"I now have comprehensive clinical data from {tool_count} analytical tools ({', '.join(tool_chain)}). I can synthesize these findings into a cohesive clinical assessment.",
            
            "integrated_summary": f"The {tool_count}-tool analysis using {', '.join(tool_chain)} provides complementary information that I can integrate for a complete diagnostic picture.",
            
            "diagnostic_synthesis": f"With results from {', '.join(tool_chain)}, I can now perform diagnostic synthesis to provide definitive conclusions about this case.",
            
            "educational_explanation": f"The multi-tool approach using {', '.join(tool_chain)} demonstrates how different analytical methods contribute to comprehensive diagnosis. I can now explain the complete findings.",
            
            "structured_findings": f"Having completed analysis with {tool_count} tools ({', '.join(tool_chain)}), I can now present structured findings addressing all aspects of the clinical question.",
            
            "narrative_assessment": f"The comprehensive evaluation using {', '.join(tool_chain)} allows me to provide a narrative assessment that tells the complete diagnostic story."
        }
        
        return thoughts_templates[style]
        
    def _generate_integrated_response(self, tool_chain: List[str], examples: List[ToolExample], style: str) -> str:
        """Generate diverse integrated responses based on style."""
        
        if style == "clinical_report":
            if len(examples) == 2:
                response = f"CLINICAL FINDINGS: {examples[0].assistant_response} ADDITIONAL ASSESSMENT: {examples[1].assistant_response} CLINICAL IMPRESSION: The combined analysis provides comprehensive diagnostic information for clinical decision-making."
            else:
                parts = [f"ANALYSIS {i+1}: {ex.assistant_response}" for i, ex in enumerate(examples)]
                response = " ".join(parts) + " SUMMARY: Multi-modal analysis complete."
                
        elif style == "integrated_summary":
            connectors = ["Building on this,", "Furthermore,", "Additionally,", "In conjunction with this,", "Complementing these findings,"]
            if len(examples) == 2:
                response = f"My comprehensive assessment reveals: {examples[0].assistant_response} {random.choice(connectors)} {examples[1].assistant_response} These integrated findings provide a complete diagnostic picture."
            else:
                parts = []
                for i, ex in enumerate(examples):
                    if i == 0:
                        parts.append(f"Initial analysis shows: {ex.assistant_response}")
                    elif i == len(examples) - 1:
                        parts.append(f"Final assessment confirms: {ex.assistant_response}")
                    else:
                        parts.append(f"{random.choice(connectors)} {ex.assistant_response}")
                response = " ".join(parts)
                
        elif style == "diagnostic_synthesis":
            if len(examples) == 2:
                response = f"DIAGNOSTIC SYNTHESIS: Combining findings from multiple analytical approaches: First, {examples[0].assistant_response} Second, {examples[1].assistant_response} CONCLUSION: The convergent evidence supports a comprehensive diagnostic assessment."
            else:
                response = f"MULTI-TOOL DIAGNOSTIC SYNTHESIS: " + " → ".join([f"{ex.assistant_response}" for ex in examples]) + " The synthesized evidence provides definitive diagnostic clarity."
                
        elif style == "educational_explanation":
            if len(examples) == 2:
                response = f"Let me walk you through the findings: Starting with the first analysis: {examples[0].assistant_response} Moving to the second evaluation: {examples[1].assistant_response} This step-by-step approach demonstrates how multiple tools provide complementary diagnostic information."
            else:
                parts = [f"Step {i+1} reveals: {ex.assistant_response}" for i, ex in enumerate(examples, 1)]
                response = "Educational walkthrough: " + " ".join(parts) + " This systematic approach showcases comprehensive diagnostic methodology."
                
        elif style == "structured_findings":
            if len(examples) == 2:
                response = f"STRUCTURED ASSESSMENT:\n• Primary Analysis: {examples[0].assistant_response}\n• Secondary Analysis: {examples[1].assistant_response}\n• Integrated Conclusion: Multi-tool evaluation provides comprehensive diagnostic clarity."
            else:
                bullet_points = [f"• Analysis {i+1}: {ex.assistant_response}" for i, ex in enumerate(examples, 1)]
                response = "COMPREHENSIVE FINDINGS:\n" + "\n".join(bullet_points) + "\n• OVERALL ASSESSMENT: Complete multi-modal diagnostic evaluation achieved."
                
        else:  # narrative_assessment
            narrative_transitions = ["The diagnostic story unfolds as follows:", "This case presents an interesting narrative:", "The complete diagnostic picture emerges:"]
            transition_words = ["Subsequently,", "Following this,", "The investigation continues with", "Further analysis reveals"]
            
            if len(examples) == 2:
                response = f"{random.choice(narrative_transitions)} {examples[0].assistant_response} {random.choice(transition_words)} {examples[1].assistant_response} This comprehensive narrative provides complete diagnostic insight."
            else:
                parts = [examples[0].assistant_response]
                for ex in examples[1:]:
                    parts.append(f"{random.choice(transition_words)} {ex.assistant_response}")
                response = f"{random.choice(narrative_transitions)} " + " ".join(parts) + " The complete diagnostic narrative is now established."
                
        return response

# -----------------------------------------------------------------------------
# Main generation function
# -----------------------------------------------------------------------------

def generate_multi_tool_conversation(
    registry: ToolRegistry, 
    extractor: RealDataExtractor
) -> Dict[str, Any]:
    """Generate a single multi-tool conversation."""
    planner = MultiToolPlanner(registry, extractor)
    builder = MultiToolConversationBuilder(registry, extractor)
    
    # Plan tool combination
    tool_chain = planner.plan_multi_tool_combination()
    if not tool_chain:
        return {}
        
    # Build conversation
    conversation = builder.build_multi_tool_conversation(tool_chain)
    return conversation

# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multi-tool single-round conversations")
    parser.add_argument("--tool_yaml", type=str, required=True, help="Path to tool metadata YAML")
    parser.add_argument("--single_round_dir", type=str, required=True, help="Path to single-round examples directory")
    parser.add_argument("--out", type=str, required=True, help="Output file path")
    parser.add_argument("--num", type=int, required=True, help="Number of conversations to generate")
    args = parser.parse_args()

    registry = ToolRegistry(args.tool_yaml)
    extractor = RealDataExtractor(args.single_round_dir)

    conversations = []
    for _ in range(args.num):
        conversation = generate_multi_tool_conversation(registry, extractor)
        if conversation:
            conversations.append(conversation)

    with open(args.out, "w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + "\n")

    print(f"Generated {len(conversations)} multi-tool conversations and saved to {args.out}")

if __name__ == "__main__":
    main()
