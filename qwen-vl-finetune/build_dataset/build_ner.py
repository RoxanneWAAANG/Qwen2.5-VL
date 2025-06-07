import json
import random
from tqdm import tqdm

INPUT_FILE   = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/RaTE-NER/train_span.json"
OUTPUT_FILE  = "./tool_instruct/rate_ner_dataset.jsonl"
MAX_SAMPLES  = 5000

instruction_templates = [
    "Perform medical named-entity recognition on the following radiology note. Label each token with its entity type (e.g., Anatomy, Abnormality, Disease):",
    "Extract and classify all medical entities in this radiology report. For each token, specify whether it's Anatomy, Abnormality, or Disease:",
    "Identify anatomical structures, abnormalities, and diseases in the text below. Return a token-level annotation indicating the entity type:",
    "What medical entities are in this phrase?",
    "Tag this short clinical description with entity types.",
    "Can you identify any Anatomy, Abnormality, or Disease terms here?",
    "Which words in this sentence are medical entities?",
    "Is there any disease or anatomy mentioned in this phrase?",
    "Mark up each word with its entity class.",
    "Highlight the medical terms in this sentence.",
    "Give me a token-level tag for this snippet.",
    "Annotate the following text with medical entity tags.",
    "Show which tokens are anatomy, abnormalities, or diseases in this sample.",
    "Help me understand entity types in this short report.",
    "Label the medical categories of words in this example sentence.",
    "Can you annotate this phrase for medical NLP training?",
    "Review this sentence and tag all medical terms by type.",
    "Identify token-level entities in this clinical line.",
    "Break down this sentence into entity-labeled tokens.",
    "Do a quick NER pass on this short clinical description.",
    "Mark entities in this expression as Anatomy/Abnormality/Disease.",
    "Tag this input using medical named-entity recognition.",
    "Can you tell what's anatomy or disease in this line?",
    "Mark each word in this short note with its medical meaning.",
    "Scan this sentence for anatomy, diseases, or abnormalities.",
    "Go through this line and label any clinical entities.",
    "Assign entity types to the words in this phrase.",
    "Quickly tag medical terms in this snippet.",
    "Label all words that are clinical entities: Anatomy, Abnormality, Disease.",
    "For each word, give a label: Anatomy, Abnormality, or Disease.",
    "Identify and tag any relevant medical terms.",
    "Please classify tokens by medical entity type.",
    "Detect entity types in the following sentence.",
    "Tag this entry for Anatomy, Abnormality, or Disease.",
    "Classify words in this input as clinical entities.",
    "Run a quick entity classification on this phrase.",
    "Perform a light NER tagging on this sentence.",
    "Classify each term as anatomy, abnormality, or disease.",
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

LABEL_MAP = {"Anatomy": "Anatomy", "Abnormality": "Abnormality", "Disease": "Disease"}

def spans_to_entities(tokens, spans):
    """
    Convert span indices to a readable bullet list: “token(s) → Label”.
    Span format in dataset: [start, end, label]
    """
    parts = []
    for start, end, label in spans:
        text = " ".join(tokens[start : end + 1])
        parts.append(f"• {text} → {LABEL_MAP.get(label, label)}")
    return "\n".join(parts) if parts else "• No entities detected"

def transform(record, idx):
    tokens = record["sentences"][0]
    spans  = record["ner"][0]

    prompt = random.choice(instruction_templates)
    human  = {
        "from": "human",
        "value": f"{prompt}\n\n{' '.join(tokens)}"
    }

    gpt_tool_call = {
        "from": "gpt",
        "thoughts": "To extract the medical entities, I'll call the RaTE-NER tool.",
        "actions": [{
            "API_name": "RaTE-NER",
            "API_params": {"tokens": tokens}
        }],
        "value": "Calling RaTE-NER to extract entities..."
    }

    pretty_entities = spans_to_entities(tokens, spans)
    friendly_reply  = random.choice(answer_templates).format(entities=pretty_entities)
    
    gpt_final_response = {
        "from": "gpt",
        "value": friendly_reply
    }

    return {
        "id": f"ner_sample_{idx}",
        "conversations": [human, gpt_tool_call, gpt_final_response]
    }

def build_dataset(input_path, output_path, max_samples=MAX_SAMPLES):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(tqdm(fin, desc="Building NER instruction data")):
            if idx >= max_samples:
                break
            record = json.loads(line)
            conv = transform(record, idx)
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Wrote {min(idx + 1, max_samples)} examples to '{output_path}'")

if __name__ == "__main__":
    build_dataset(INPUT_FILE, OUTPUT_FILE)
