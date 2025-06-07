import json, random, xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob
from tqdm import tqdm

INPUT_FILE = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/pubmedqa/pubmedqa.json"
OUTPUT_FILE = "./tool_instruct/pmc_llama_medqa_dataset.jsonl"
MAX_SAMPLES = 10000

prompt_templates = [
    "Can you answer this medical question based on the text provided?",
    "Read the passages and help answer this biomedical question clearly.",
    "What's the best answer using the info in these excerpts?",
    "Give me a reasoned answer using the info shown.",
    "From what you see here, how would you answer this medical question?",
    "Take a look at the evidence. What's the correct answer and why?",
    "Based on this info, what's your conclusion?",
    "Use these research snippets to figure out the answer.",
    "Let's solve this question using the given evidence.",
    "Summarize the answer based on the details in the literature.",
    "Answer the biomedical question using the following study excerpts.",
    "Rely on the provided texts to answer the question accurately.",
    "Use the passages below to construct a correct answer with a brief explanation.",
    "Refer to the provided abstracts and generate a complete, supported answer.",
    "Support your answer with information drawn exclusively from the literature below.",
    "This question tests your understanding of the given evidence. Answer and explain.",
    "Treat this as a research QA task. Answer using cited evidence.",
    "Examine the snippets, then write a correct and concise response.",
    "Your goal: produce an accurate answer supported by textual evidence.",
    "Read, reason, and respond based on the provided biomedical excerpts.",
    "Analyze the excerpts and respond with an evidence-based conclusion.",
    "Generate an answer to the medical query, explaining briefly how you got there.",
    "Use your understanding of the text to answer clearly and concisely.",
    "Derive the answer from the evidence, then explain your logic.",
    "Support your answer with a rationale rooted in the provided information.",
    "Read the literature, answer the question.",
    "Evaluate the context and draw a grounded conclusion.",
    "Develop your answer from these research fragments and explain briefly.",
    "Synthesize an answer from the data and outline your thought process.",
    "Deliver answer and follow it with reasoning based on the source.",
    "Answer the question using the evidence.",
    "Use the snippets to form a concise, accurate response.",
    "From these texts, give a supported answer.",
    "What does the evidence suggest? Provide your answer.",
    "Create an answer with a short explanation using the text.",
    "Use the paragraphs to answer and summarize your reasoning.",
    "Refer to this content to answer the biomedical query.",
    "Read the snippets and give your answer with rationale.",
    "Use the following to explain and answer the question.",
    "Your task: answer based on these passages alone.",
    "I'll show you a question and some references. Please respond with a justified answer.",
    "Here's a question with relevant literature. What's your answer?",
    "I'm giving you evidence for a biomedical query. How would you answer it?",
    "Using this information, help answer the question and explain your logic.",
    "Review the texts and respond with reasoning.",
    "Interpret the question and answer clearly.",
    "As an assistant, respond to the query with evidence-grounded reasoning.",
    "Help draft an evidence-backed response for this health-related question.",
]

answer_templates = [
    "Here's a concise answer based on the analysis:\n{answer}",
    "Answer and reasoning:\n{answer}",
    "This is the requested information:\n{answer}",
    "Below is the detailed response:\n{answer}",
    "I've completed the review—see the answer here:\n{answer}",
    "Final answer with explanation:\n{answer}",
    "Here is the answer in full:\n{answer}",
    "Here is the solution along with the rationale:\n{answer}",
    "Comprehensive answer:\n{answer}",
    "Detailed answer provided below:\n{answer}",
    "Here's what the evidence indicates:\n{answer}",
    "The question is addressed as follows:\n{answer}",
    "Answer (including supporting details):\n{answer}",
    "Full response:\n{answer}",
    "My complete answer is:\n{answer}",
    "The findings are summarized here:\n{answer}",
    "Answer, with relevant details:\n{answer}",
    "Completed answer:\n{answer}",
    "Response with explanation:\n{answer}",
    "Full explanation and answer:\n{answer}",
    "Here is the final response:\n{answer}",
    "Solution and rationale:\n{answer}",
    "Comprehensive explanation:\n{answer}",
    "The final answer is presented below:\n{answer}",
    "After analysis, the answer is:\n{answer}",
    "Full answer (see details):\n{answer}",
    "My conclusion:\n{answer}",
    "Please find the answer here:\n{answer}",
    "Answer with supporting points:\n{answer}",
    "Here is an in-depth answer:\n{answer}",
    "Explicit answer and reasoning:\n{answer}",
    "Definitive answer:\n{answer}",
    "Answer summary:\n{answer}",
    "Complete solution:\n{answer}",
    "Here's the resolved answer:\n{answer}",
    "The following addresses your query:\n{answer}",
    "Answer (detailed):\n{answer}",
    "Here's the explanation and answer:\n{answer}",
    "My detailed response:\n{answer}",
    "Answer provided below:\n{answer}",
]

def join_context(ctx):
    return "\n\n".join(ctx) if isinstance(ctx, list) else str(ctx)

def transform(ex, idx):
    question = ex["question"].strip()
    context  = join_context(ex["context"]["contexts"])
    answer   = ex.get("long_answer", "").strip()

    system_prompt = random.choice(prompt_templates)
    user_prompt   = (
        system_prompt +
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context}"
    )

    friendly_reply = random.choice(answer_templates).format(answer=answer)

    return {
        "id": f"pubmedqa_{idx}",
        "conversations": [
            {
                "from": "human", 
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": "To answer this question and provide a detailed rationale, I'll call the PMC-LLaMA model.",
                "actions": [
                    {
                        "API_name": "PMC-LLaMA",
                        "API_params": {"query": question}
                    }
                ],
                "value": "Calling PMC-LLaMA to get the answer and rationale..."
            },
            {
                "from": "gpt", 
                "value": friendly_reply
            }
        ]
    }

def build_instruction_dataset(input_path, output_path, max_samples):
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    subset = examples[:max_samples]

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, ex in enumerate(tqdm(subset, desc=f"Transforming first {max_samples} examples")):
            record = transform(ex, idx)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(subset)} records to '{output_path}'")

if __name__ == "__main__":
    build_instruction_dataset(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES)
