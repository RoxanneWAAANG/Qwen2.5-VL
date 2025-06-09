import json
import random
from tqdm import tqdm

INPUT_FILE = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/RaTE-NER/train_span.json"
OUTPUT_FILE = "./tool_instruct/rate_ner_dataset.jsonl"
IMAGE_BASE_PATH = "/home/jack/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune/build_dataset/dummy_images"
DUMMY_IMAGE_NAME = "dummy_img.png"
MAX_SAMPLES = 5000

prompt_templates = [
    "<image>\nPlease identify and label all medical entities in the following text.",
    "<image>\nCan you extract and classify the medical entities from this clinical note?",
    "<image>\nHelp me identify the anatomical terms, abnormalities, and diseases in this text.",
    "<image>\nPlease perform named entity recognition on this medical sentence.",
    "<image>\nIdentify all medical entities and their types in the given text.",
    "<image>\nExtract and categorize the medical terms from this clinical sentence.",
    "<image>\nCan you label the entities in this medical text with their appropriate categories?",
    "<image>\nPlease recognize and classify all medical entities present in this sentence.",
    "<image>\nIdentify medical entities including anatomy, abnormalities, and diseases.",
    "<image>\nPerform entity extraction on this clinical text.",
    "<image>\nLabel all medical terms with their entity types in this sentence.",
    "<image>\nCan you detect and classify medical entities from this clinical note?",
    "<image>\nHelp me tag all medical entities in this text with their categories.",
    "<image>\nPlease identify medical entities and assign appropriate labels.",
    "<image>\nExtract named entities from this medical sentence and classify them.",
    "<image>\nRecognize and categorize all medical terms in the following text.",
    "<image>\nCan you perform medical NER on this clinical sentence?",
    "<image>\nIdentify and label medical entities including anatomical and pathological terms.",
    "<image>\nPlease extract medical entities and their classifications from this text.",
    "<image>\nTag all relevant medical entities in this clinical sentence.",
    "<image>\nCan you identify medical terms and classify them by entity type?",
    "<image>\nPerform named entity recognition focusing on medical terminology.",
    "<image>\nLabel medical entities in this text with anatomy, abnormality, or disease tags.",
    "<image>\nExtract and classify medical entities from this clinical documentation.",
    "<image>\nPlease identify all medical entities and provide their entity types.",
    "<image>\nRecognize medical terms and categorize them appropriately.",
    "<image>\nCan you detect medical entities and assign proper classifications?",
    "<image>\nHelp me label medical entities in this clinical text.",
    "<image>\nIdentify and categorize medical terminology in this sentence.",
    "<image>\nPerform medical entity extraction and classification on this text.",
    "<image>\nPlease tag medical entities with their corresponding categories.",
    "<image>\nExtract named entities focusing on medical and clinical terms.",
    "<image>\nCan you identify and classify medical entities in this clinical note?",
    "<image>\nLabel all medical terms with appropriate entity classifications.",
    "<image>\nRecognize and categorize medical entities from this text.",
    "<image>\nPlease perform entity recognition on this medical sentence.",
    "<image>\nIdentify medical entities and assign them to proper categories.",
    "<image>\nExtract and label medical terms from this clinical text.",
    "<image>\nCan you detect and classify all medical entities present?",
    "<image>\nHelp me identify and categorize medical terminology in this sentence.",
]

answer_templates = [
    "Here is the entity breakdown:\n{entities}",
    "Below are the entities I identified:\n{entities}",
    "These are the tokens and their corresponding categories:\n{entities}",
    "Entities detected:\n{entities}",
    "Here are the labeled medical terms:\n{entities}",
    "The note contains the following entities:\n{entities}",
    "Token-level annotations are as follows:\n{entities}",
    "I found these entities in the report:\n{entities}",
    "Entity tagging results:\n{entities}",
    "Summary of entities:\n{entities}",
    "I've categorized the tokens like this:\n{entities}",
    "Entities extracted from the sentence:\n{entities}",
    "Here is the complete list of entities:\n{entities}",
    "Annotated entities:\n{entities}",
    "Below is the entity list with labels:\n{entities}",
    "Findings — entities by type:\n{entities}",
    "This is the entity mapping:\n{entities}",
    "Token annotations:\n{entities}",
    "Here are the recognized entities:\n{entities}",
    "Entity extraction complete:\n{entities}",
    "The following entities were detected:\n{entities}",
    "Entity recognition output:\n{entities}",
    "I've highlighted each entity below:\n{entities}",
    "Detailed entity list:\n{entities}",
    "Here are the classified terms:\n{entities}",
    "Entities present in the text:\n{entities}",
    "Medical entity labels:\n{entities}",
    "Here's a breakdown of the entities found:\n{entities}",
    "Identified entities:\n{entities}",
    "Entity results:\n{entities}",
    "These terms have been labeled:\n{entities}",
    "Entity analysis:\n{entities}",
    "The detected entities are listed below:\n{entities}",
    "I've listed all entities with their labels:\n{entities}",
    "Entity report:\n{entities}",
    "Complete entity annotation:\n{entities}",
    "Here is the token classification:\n{entities}",
    "Entities and their types:\n{entities}",
    "Recognized medical entities:\n{entities}",
    "Token-by-token entity mapping:\n{entities}",
]

def generate_image_id():
    """Generate a medical image ID for format compatibility
    
    Even though we're using dummy images, we maintain realistic ID format
    to ensure compatibility with systems expecting medical image identifiers
    """
    segments = []
    for _ in range(5):
        segment = format(random.randint(0, 0xffffffff), '08x')
        segments.append(segment)
    return '-'.join(segments)

def create_image_filename(image_id):
    """Create dummy image filename and full path
    
    Uses a single dummy image for all entries since this is a text-based NER task.
    The image_id is still unique for each record to maintain proper format structure.
    """
    full_path = f"{IMAGE_BASE_PATH}/{DUMMY_IMAGE_NAME}"
    return DUMMY_IMAGE_NAME, full_path

def reconstruct_sentence(tokens):
    """Reconstruct sentence from token list
    
    Takes a list of tokens and joins them into a readable sentence.
    This handles the tokenized format of your NER dataset.
    """
    return " ".join(tokens)

def format_entities(tokens, entity_annotations):
    """Format entity annotations for display
    
    Converts the span-based entity annotations into human-readable format
    showing which tokens correspond to which entity types.
    """
    if not entity_annotations or not entity_annotations[0]:
        return "No entities found."
    
    entity_list = []
    for start, end, entity_type in entity_annotations[0]:  # First sentence annotations
        # Extract the entity tokens using the span indices
        entity_tokens = tokens[start:end+1]
        entity_text = " ".join(entity_tokens)
        entity_list.append(f"'{entity_text}' -> {entity_type}")
    
    return "\n".join(entity_list)

def transform(ex, idx):
    """Transform a NER example into the new image-based format while preserving NER functionality"""
    
    note_id = ex["note_id"]
    sentences = ex["sentences"]
    ner_annotations = ex["ner"]

    sentence_tokens = sentences[0] if sentences else []
    sentence_text = reconstruct_sentence(sentence_tokens)

    image_id = generate_image_id()
    image_filename, full_path = create_image_filename(image_id)
    
    system_prompt = random.choice(prompt_templates)
    
    user_prompt = (
        system_prompt +
        f"\n\n### Text to analyze:\n{sentence_text}"
    )
    
    formatted_entities = format_entities(sentence_tokens, ner_annotations)
    
    ner_output = f"RaTE-NER output: {formatted_entities}\n\nAnswer my first request: {system_prompt.replace('<image>', '').strip()}\n\n"

    friendly_reply = random.choice(answer_templates).format(entities=formatted_entities)
    
    return {
        "image_id": image_id,
        "image": image_filename,
        "file_name": full_path,
        "conversations": [
            {
                "from": "human",
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": "To perform named entity recognition on this text, I'll call the RaTE-NER model.",
                "actions": [
                    {
                        "API_name": "RaTE-NER",
                        "API_params": {"text": sentence_text}
                    }
                ],
                "value": "Calling RaTE-NER tool to identify and classify medical entities..."
            },
            {
                "from": "human",
                "value": ner_output
            },
            {
                "from": "gpt",
                "thoughts": "The RaTE-NER has completed the entity recognition. Now I can provide the formatted results.",
                "actions": [],
                "value": friendly_reply
            }
        ]
    }

def build_instruction_dataset(input_path, output_path, max_samples):
    """Build the medical NER dataset in the new image-based format"""
    
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num + 1}: {e}")
                    continue
    
    print(f"Loaded {len(examples)} examples from {input_path}")
    
    subset = examples[:max_samples]
    
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, ex in enumerate(tqdm(subset, desc=f"Transforming first {max_samples} NER examples")):
            record = transform(ex, idx)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nWrote {len(subset)} NER records to '{output_path}'")
    print(f"All records use dummy image: {DUMMY_IMAGE_NAME}")

if __name__ == "__main__":
    build_instruction_dataset(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES)
