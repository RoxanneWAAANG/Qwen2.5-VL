import random
import json
import yaml
import argparse
import uuid
from typing import Dict, List

def load_lines(filepath: str) -> List[str]:
    """Load lines from a text file"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# ============================================================================
# TEMPLATE FOR Multi-Round Sessions
# ============================================================================

# Probability to prefix a chat-only round before the tool rounds
PREFIX_CHAT_PROB = 0.2

# 1. INITIAL REQUEST STYLES
INITIAL_REQUEST_STYLES = [
    # Professional/Clinical
    "Please {action}",
    "I need you to {action}",
    "Can you {action}",
    "Could you please {action}",
    "Please proceed with {action}",
    "I require {action}",
    "Kindly {action}",
    "Would you mind {action}",
    "I'd like you to {action}",
    "Please help me {action}",
    
    # Conversational/Casual
    "Hi! Could you help me {action}?",
    "Hello, can you {action}?",
    "Hey there! I need help with {action}",
    "Mind helping me {action}?",
    "Could you give me a hand and {action}?",
    "Hi! I'm looking to {action}",
    "Hello! Would you be able to {action}?",
    "Hey! Can you please {action}?",
    "Hi there! Could you {action}?",
    "Good morning! Please {action}",
    "Good afternoon! Can you {action}?",
    "Greetings! I need you to {action}",
    
    # Urgent/Clinical
    "This is urgent - please {action}",
    "Emergency case: {action} needed",
    "Time-sensitive: please {action} quickly",
    "Urgent: patient needs {action}",
    "STAT {action} required",
    "Immediate {action} needed",
    "Priority case - {action} requested",
    "Critical: please {action} now",
    "Emergency department needs {action}",
    "Rapid {action} required",
    
    # Educational/Learning
    "I'm learning - can you {action} and explain?",
    "For educational purposes, please {action}",
    "Help me understand by {action}",
    "Could you demonstrate how to {action}?",
    "Show me the proper way to {action}",
    "I'm a student - please {action} and teach me",
    "For training purposes, {action}",
    "As a learning exercise, {action}",
    "Educational demonstration: {action}",
    "Teaching case: please {action}",
    
    # Research/Academic
    "For research purposes, please {action}",
    "Study protocol requires {action}",
    "Data collection needs {action}",
    "Research validation: {action}",
    "Clinical trial requires {action}",
    "For our study, please {action}",
    "Academic research needs {action}",
    "Scientific analysis: {action}",
    "Research methodology requires {action}",
    "Systematic study: {action}",
    
    # Consultation/Second Opinion
    "I need a second opinion - {action}",
    "Please verify by {action}",
    "Confirm my assessment: {action}",
    "Consultant input needed: {action}",
    "Expert opinion required: {action}",
    "Specialist consultation: {action}",
    "Peer review: please {action}",
    "Quality assurance: {action}",
    "Double-check by {action}",
    "Validation needed: {action}"
]

# 2. FOLLOWUP REQUEST PATTERNS
FOLLOWUP_PATTERNS = [
    # Refinement requests
    "The result looks good, but can you focus more on the {area}?",
    "Could you enhance the {aspect} in your analysis?",
    "Please refine the output around the {region}",
    "Can you improve the quality in the {zone}?",
    "Fine-tune the analysis for better {outcome}",
    "Adjust the focus to the {target}",
    "Can you sharpen the {feature}?",
    "Please optimize the {parameter}",
    "Enhance the visibility of {structure}",
    "Improve the definition of {boundary}",
    "Refine the details in {location}",
    "Polish the analysis of {component}",
    "Perfect the visualization of {element}",
    "Sharpen the focus on {anomaly}",
    "Clarify the {finding}",
    
    # Parameter adjustments
    "Can you redo this with higher sensitivity?",
    "Try again with different parameters",
    "Adjust the settings for better {metric}",
    "Modify the approach for this case",
    "Optimize the parameters",
    "Increase the resolution",
    "Enhance the contrast",
    "Boost the signal strength",
    "Amplify the {characteristic}",
    "Intensify the {property}",
    "Strengthen the {signal}",
    "Maximize the {quality}",
    "Calibrate for {condition}",
    "Tune for optimal {result}",
    "Configure for enhanced {output}",
    
    # Detail exploration
    "Great! Now focus on the {area} in more detail",
    "Zoom in on the {finding} you identified",
    "Examine the {region} more closely",
    "Provide detailed analysis of the {structure}",
    "Can you investigate the {anomaly} further?",
    "Drill down into the {component}",
    "Explore the {feature} more thoroughly",
    "Scrutinize the {element} carefully",
    "Inspect the {zone} meticulously",
    "Analyze the {boundary} precisely",
    "Study the {pattern} in detail",
    "Examine the {texture} closely",
    "Investigate the {density} variations",
    "Assess the {morphology} carefully",
    "Evaluate the {characteristics} thoroughly",
    
    # Comparative analysis
    "Now try the same analysis with a different approach",
    "Can you repeat this using an alternative method?",
    "Compare this with a more conservative approach",
    "Apply a different technique",
    "Use an alternative strategy",
    "Try a secondary method",
    "Implement a backup approach",
    "Execute using different algorithm",
    "Process with alternative parameters",
    "Analyze using complementary technique",
    "Cross-validate with different method",
    "Verify using alternate approach",
    "Confirm with secondary analysis",
    "Double-check with different settings",
    "Validate using alternative protocol",
    
    # Technical depth requests
    "Can you provide more technical details?",
    "I need a more comprehensive breakdown",
    "Please include advanced parameters",
    "Provide expert-level interpretation",
    "Give me the full technical analysis",
    "Include quantitative measurements",
    "Add statistical analysis",
    "Provide numerical data",
    "Include confidence intervals",
    "Add uncertainty estimates",
    "Provide technical specifications",
    "Include diagnostic metrics",
    "Add performance indicators",
    "Provide accuracy measurements",
    "Include quality assessments",
    
    # Clinical context
    "How does this relate to the patient's condition?",
    "What are the clinical implications?",
    "Can you provide diagnostic significance?",
    "What treatment considerations apply?",
    "How urgent is this finding?",
    "What follow-up is recommended?",
    "Are there any red flags?",
    "What's the differential diagnosis?",
    "How confident are you in this assessment?",
    "What additional tests might be needed?",
    "Is this within normal limits?",
    "What pathology is suggested?",
    "How does this compare to normal?",
    "What monitoring is required?",
    "What's the clinical significance?",
    
    # Workflow continuations
    "Excellent! Now let's move to the next step",
    "Perfect! Can you also check for {condition}?",
    "Great work! Additionally, please examine {feature}",
    "Good! Now analyze the {component} as well",
    "Wonderful! Please also assess {parameter}",
    "Fantastic! Include evaluation of {element}",
    "Superb! Now investigate {aspect} too",
    "Outstanding! Please review {characteristic}",
    "Brilliant! Also consider {factor}",
    "Impressive! Additionally evaluate {metric}"
]

# 3. MEDICAL VOCABULARY FOR SUBSTITUTION
MEDICAL_VARIABLES = {
    "area": ["lung fields", "cardiac silhouette", "mediastinum", "periphery", "central zone", 
             "upper lobe", "lower lobe", "right side", "left side", "bilateral areas",
             "apical region", "basal segments", "hilum", "costophrenic angles", "diaphragm",
             "chest wall", "pleural space", "retrocardiac space", "paratracheal region",
             "subcarinal space", "aortic arch", "pulmonary vessels", "bronchi", "parenchyma"],
    
    "aspect": ["contrast", "resolution", "clarity", "sharpness", "definition", "brightness",
               "intensity", "saturation", "detail level", "texture", "edges", "boundaries",
               "gradients", "shadows", "highlights", "noise reduction", "enhancement",
               "filtering", "smoothing", "sharpening"],
    
    "region": ["thoracic cavity", "abdominal region", "pelvic area", "cranial space",
               "spinal column", "extremities", "joint spaces", "soft tissues", "bone structures",
               "vascular system", "lymphatic system", "respiratory tract", "gastrointestinal tract",
               "urogenital system", "nervous system", "muscular system", "skeletal system"],
    
    "zone": ["anterior", "posterior", "lateral", "medial", "superior", "inferior",
             "proximal", "distal", "superficial", "deep", "internal", "external",
             "ventral", "dorsal", "cephalic", "caudal", "peripheral", "central"],
    
    "outcome": ["diagnostic accuracy", "image quality", "visualization", "detection rate",
                "sensitivity", "specificity", "precision", "recall", "confidence",
                "reliability", "consistency", "reproducibility", "validity"],
    
    "target": ["lesion", "mass", "nodule", "opacity", "density", "abnormality",
               "pathology", "finding", "structure", "organ", "tissue", "vessel"],
    
    "feature": ["texture pattern", "edge definition", "signal intensity", "density variation",
                "morphological characteristics", "anatomical landmarks", "pathological signs",
                "normal variants", "artifacts", "noise patterns"],
    
    "structure": ["organ system", "anatomical structure", "tissue architecture", "cellular organization",
                  "vascular network", "neural pathways", "skeletal framework", "muscular arrangement"],
    
    "finding": ["abnormal shadow", "increased density", "decreased attenuation", "irregular border",
                "mass effect", "air bronchogram", "consolidation", "infiltrate", "effusion",
                "pneumothorax", "cardiomegaly", "atelectasis"],
    
    "condition": ["inflammation", "infection", "malignancy", "trauma", "degeneration",
                  "congenital anomaly", "vascular disease", "metabolic disorder"]
}

# 4. CONVERSATION STARTERS
CONVERSATION_STARTERS = [
    "Hi! ", "Hello! ", "Hey there! ", "Good morning! ", "Good afternoon! ", "Greetings! ",
    "Hi there! ", "Hello there! ", "Hey! ", "Good day! ", "", "", "", "", "", "",
    "Doctor, ", "Can you help? ", "I have a question - ", "Quick question - ",
    "Excuse me, ", "Pardon me, ", "If you don't mind, ", "When you have a moment, "
]

# 5. TRANSITION PHRASES
TRANSITION_PHRASES = [
    "Excellent work! ", "Perfect! ", "Great! ", "Wonderful! ", "Fantastic! ", "Outstanding! ",
    "Brilliant! ", "Superb! ", "Impressive! ", "Remarkable! ", "Amazing! ", "Terrific! ",
    "Good! ", "Very good! ", "Well done! ", "Nice work! ", "Great job! ", "Excellent! ",
    "That's helpful! ", "That's perfect! ", "That's exactly what I needed! ", "Spot on! ",
    "That looks good, ", "That's a good start, ", "I can see that, ", "Interesting, ",
    "I notice that, ", "Looking at this, ", "Based on that, ", "Given these results, ",
    "Now that I see this, ", "With this information, ", "Taking this into account, ",
    "Considering these findings, ", "Given this analysis, ", "Based on your assessment, ",
    "Following up on that, ", "Building on that, ", "Additionally, ", "Furthermore, ",
    "Moreover, ", "Also, ", "In addition, ", "Next, ", "Then, ", "Subsequently, "
]

# 6. COMPLETION RESPONSES
COMPLETION_RESPONSES = [
    "The {task} analysis has been completed successfully.",
    "I've finished the {task} as requested.",
    "The {task} procedure is now complete.",
    "Your {task} analysis is ready.",
    "The {task} has been processed successfully.",
    "I've completed the {task} evaluation.",
    "The {task} assessment is finished.",
    "Your {task} request has been fulfilled.",
    "The {task} analysis shows the results.",
    "I've processed your {task} request.",
    "The {task} examination is complete.",
    "Your {task} analysis has been generated.",
    "The {task} study is now available.",
    "I've completed the {task} review.",
    "The {task} interpretation is ready.",
    "Your {task} evaluation is finished.",
    "The enhanced {task} analysis is complete.",
    "I've refined the {task} results.",
    "The improved {task} is now available.",
    "The optimized {task} has been generated.",
    "The detailed {task} analysis is ready.",
    "The comprehensive {task} is complete.",
    "The advanced {task} evaluation is finished.",
    "The thorough {task} assessment is done.",
    "The in-depth {task} analysis is available.",
    "Here's your completed {task}.",
    "The {task} results are now ready.",
    "Your {task} has been successfully processed.",
    "The {task} output is available for review.",
    "I've generated the {task} you requested.",
    "The {task} workflow is now complete.",
    "Your {task} analysis meets the requirements.",
    "The {task} has been executed successfully.",
    "The {task} protocol has been completed.",
    "Your customized {task} is ready.",
    "The specialized {task} analysis is done.",
    "The targeted {task} evaluation is complete.",
    "The focused {task} assessment is finished.",
    "The precision {task} analysis is available.",
    "The high-quality {task} results are ready."
]

# 7. SMALL-TALK TEMPLATES
SMALL_TALK_USER = [
    "By the way, how's your day going?", "How are you today?", "Hope you're doing well."
]
SMALL_TALK_AGENT = [
    "I'm doing well, thanks!", "All good here—ready when you are.", "I'm fine, thank you!"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_lines(filepath: str) -> List[str]:
    """Load non-empty lines from a text file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def enhance_request(original_request: str) -> str:
    """Wrap original request in a random starter + style."""
    clean = original_request.replace("<image>", "").strip()
    starter = random.choice(CONVERSATION_STARTERS)
    style = random.choice(INITIAL_REQUEST_STYLES)
    return f"<image>\n{starter}{style.format(action=clean)}"

def generate_followup(tool_output: str, original_request: str) -> str:
    """Generate a follow-up using transition + pattern + medical vars."""
    clean = original_request.replace("<image>", "").strip()
    transition = random.choice(TRANSITION_PHRASES)
    pattern = random.choice(FOLLOWUP_PATTERNS)
    # substitute one medical variable if present
    for var, opts in MEDICAL_VARIABLES.items():
        if "{" + var + "}" in pattern:
            pattern = pattern.format(**{var: random.choice(opts)})
    return f"{tool_output}\n{transition}{pattern}"

def generate_completion(task: str) -> str:
    """Pick a random completion template and fill in the task."""
    return random.choice(COMPLETION_RESPONSES).format(task=task)

def extract_tool_info(gpt_msg: Dict) -> (str, str):
    """Extract tool name and task from a GPT message."""
    actions = gpt_msg.get("actions", [])
    if actions:
        tool_name = actions[0].get("API_name", "HealthGPT")
        task = actions[0].get("API_params", {}).get("task", "analyze")
    else:
        tool_name, task = "HealthGPT", "analyze"
    return tool_name, task

# ============================================================================
# MAIN SESSION GENERATION
# ============================================================================

def make_refine_round(tool_name: str, meta: Dict, img_ref: str, example: Dict, followups: List[str]) -> List[Dict]:
    """Build a 4-message refine round for a given tool."""
    followup = random.choice(followups)
    # Human refine request
    human = {
        "from": "human",
        "value": f"{followup} {meta['refine_task']} on this {meta['modality']} ({img_ref})."
    }
    # Agent tool call
    agent_call = {
        "from": "gpt",
        "thoughts": "…",
        "actions": [{"API_name": tool_name, "API_params": meta["default_args"]}],
        "value": f"Calling {tool_name} to {meta['refine_task']}…"
    }
    # Simulated tool output
    tool_out = {
        "from": "human",
        "value": f"{tool_name} output: refined_{img_ref}"
    }
    # Agent refine response
    agent_resp = {
        "from": "gpt",
        "thoughts": "…",
        "actions": [],
        "value": random.choice(meta["refine_responses"])
    }
    return [human, agent_call, tool_out, agent_resp]

def generate_session(example: Dict, tool_metadata: Dict, greetings: List[str], 
                    followups: List[str]) -> Dict:
    """Generate enhanced multi-round session - SIMPLIFIED VERSION"""
    original_conv = example["conversations"]
    orig_human = original_conv[0]["value"]
    orig_gpt = original_conv[1]
    tool_name, task = extract_tool_info(orig_gpt)
    meta = tool_metadata[tool_name]
    img_ref = example["image"]
    
    rounds = []
    
    # Optionally prefix a greeting/chat-only round
    if random.random() < PREFIX_CHAT_PROB:
        rounds.append([
            {"from": "human", "value": random.choice(greetings)},
            {"from": "gpt", "thoughts": "…", "actions": [], "value": random.choice(SMALL_TALK_AGENT)},
            {"from": "human", "value": "I'm ready to proceed."},
            {"from": "gpt", "thoughts": "…", "actions": [], "value": "Sure—what would you like me to do?"}
        ])
        # Then treat the original as Round 2
        rounds.append(original_conv)
        # Finally, always add a refine round as Round 3
        rounds.append(make_refine_round(tool_name, meta, img_ref, example, followups))
    else:
        # No prefix: Round 1 = original, Round 2 = refine
        rounds.append(original_conv)
        rounds.append(make_refine_round(tool_name, meta, img_ref, example, followups))
        # Optionally add a closing small-talk as Round 3
        if random.random() < 0.3:
            rounds.append([
                {"from": "human", "value": random.choice(SMALL_TALK_USER)},
                {"from": "gpt",   "thoughts": "…", "actions": [], "value": random.choice(SMALL_TALK_AGENT)},
                {"from": "human", "value": "No, that's all—thanks!"},
                {"from": "gpt",   "thoughts": "…", "actions": [], "value": "Glad to help—take care!"}
            ])
    
    # Flatten rounds into one conversation list
    conversations = [msg for rnd in rounds for msg in rnd]
    
    return {
        "session_id": str(uuid.uuid4()),
        "image_id": example["image_id"],
        "image": example["image"],
        "file_name": example["file_name"],
        "conversations": conversations
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-round dialogues for single-tool deep-dive sessions."
    )
    parser.add_argument("--input", required=True, help="Input JSONL of single-round examples.")
    parser.add_argument("--tool_metadata", required=True, help="YAML file with tool metadata.")
    parser.add_argument("--greetings", required=True, help="TXT file of greeting phrases.")
    parser.add_argument("--followups", required=True, help="TXT file of follow-up phrases.")
    parser.add_argument("--output", required=True, help="Output JSONL for multi-round sessions.")
    parser.add_argument("--multiplier", type=int, default=1, help="Generate N variations per input example")
    args = parser.parse_args()

    # Load data
    examples = [json.loads(line) for line in open(args.input, 'r')]
    tool_metadata = yaml.safe_load(open(args.tool_metadata, 'r'))
    greetings = load_lines(args.greetings)
    followups = load_lines(args.followups)

    total_generated = 0
    
    # Generate and write sessions
    with open(args.output, 'w') as out_f:
        for ex in examples:
            # Generate multiple variations per example for 20k target
            for _ in range(args.multiplier):
                session = generate_session(ex, tool_metadata, greetings, followups)
                out_f.write(json.dumps(session) + "\n")
                total_generated += 1
    
    print(f"Generated {total_generated} diverse multi-round sessions")
    print(f"From {len(examples)} original examples with {args.multiplier}x multiplier")

if __name__ == "__main__":
    main()
