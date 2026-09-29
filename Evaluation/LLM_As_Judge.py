import os
import argparse
import time
import csv
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


SYSTEM_PROMPT = """
You will be given a model generated lay summary along with a complex judgement. A lay summary is referred to as a summary of a complex judgement, which is comprehensible and easier to understand for a reader with up to a high school(12th grade) education.

Your task is to rate the model generated lay summary on some metrics. Please make sure you read and understand these instructions carefully. Please keep these documents open while reviewing and refer to it as needed. 

Evaluation Criteria:
Relevance & Coverage (RC) : Does the summary cover most of the points from the judgement without missing key information?
	- A score of 1 means completely irrelevant or missing major points.
	- A score of 2 means covers very few relevant aspects.
	- A score of 3 means covers some points but misses several key details.
	- A score of 4 means covers most key points with minor omissions.
	- A score of 5 means fully captures important aspects of the judgement.

Factual Consistency (FC) : Is the model generated lay summary free from hallucination or false claims?
- A score of 1 means major false information and contradiction in model generated lay summary. 
- A score of 2 means presence of multiple incorrect or misleading statements.
- A score of 3 means there are some inaccuracies but the core idea is partly correct.
- A score of 4 means mostly accurate with presence of minor errors.
- A score of 5 means completely accurate and no hallucinations.

Legal Terms Explained (LTE) : Are legal terms either avoided or clearly explained in simple and easier language?
- A score of 1 means heavy use of unexplained legal jargon.
- A score of 2 means frequent legal terms with little explanation.
- A score of 3 means some legal terms, partially explained.
- A score of 4 means minimal legal jargon, mostly explained.
- A score of 5 means no legal terms  or all legal terms clearly explained in simple terms.

Grammaticality (Gr) : Is the summary grammatically correct and well formed?
- A score of 1 means very poor grammar, hard to read..
- A score of 2 means frequent grammatical errors.
- A score of 3 means some noticeable errors.
- A score of 4 means minor grammatical issues.
- A score of 5 means grammatically correct and fluent.

Coherence (Co) : Do ideas connect logically across sentences in model generated lay summary?
- A score of 1 means completely disjointed.
- A score of 2 means poor flow and hard to follow.
- A score of 3 means some logical flow but inconsistent.
- A score of 4 means mostly coherent with minor issues.
- A score of 5 means fully logical and well-structured.

Conciseness & Hallucination (CH) : Is the summary concise while avoiding unnecessary details or fabricated content?
- A score of 1 means contains hallucinations and very verbose.
- A score of 2 means too long with significant incorrect content.
- A score of 3 means some unnecessary details or minor hallucinations.
- A score of 4 means mostly concise with minimal extra content.
- A score of 5 means concise, precise and no hallucination.

Evaluation Steps:
1) Read the complex judgement and model generated lay summary.
2) Rate the model generated lay summary on a scale of 1 to 5 according to the evaluation criteria above.
3) Provide a brief explanation of your rating, referring to specific aspects of model generated lay summary and complex judgement.
"""


USER_PROMPT_TEMPLATE = """
You will be given a model generated lay summary along with a complex judgement. Your task is to rate the model generated lay summary on some metrics. Please make sure you read and understand these instructions carefully. Please keep these documents open while reviewing and refer to it as needed. 

COMPLEX JUDGEMENT:
{judgement}

MODEL-GENERATED LAY SUMMARY:
{laysummary}

Evaluation Steps:
1) Read the complex judgement and model generated lay summary.
2) Rate the model generated lay summary on a scale of 1 to 5 according to the evaluation criteria above.
3) Provide a brief explanation of your rating, referring to specific aspects of model generated lay summary and complex judgement.

Relevance & Coverage (RC): <1-5>
Factual Consistency (FC) Score: <1-5>
Legal Terms Explained (LTE) Score: <1-5>
Grammaticality (Gr) Score: <1-5>
Coherence (Co) Score: <1-5>
Conciseness & Hallucination (CH) Score: <1-5>
"""


def load_model(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    return tokenizer, model


def build_prompt(tokenizer, judgement, summary):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            judgement=judgement.strip(),
            laysummary=summary.strip()
        )}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_response(tokenizer, model, prompt, max_tokens):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=10000
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract only generated part
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return decoded.strip()


def extract_scores(text):
    patterns = {
        "RC": r"(?:RC|Relevance\s*&\s*Coverage)[^0-9]*([1-5])",
        "FC": r"(?:FC|Factual\s*Consistency)[^0-9]*([1-5])",
        "LTE": r"(?:LTE|Legal\s*Terms\s*Explained)[^0-9]*([1-5])",
        "Gr": r"(?:Gr|Grammaticality)[^0-9]*([1-5])",
        "Co": r"(?:Co|Coherence)[^0-9]*([1-5])",
        "CH": r"(?:CH|Conciseness\s*&\s*Hallucination)[^0-9]*([1-5])"
    }

    scores = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        scores[key] = int(match.group(1)) if match else None

    return scores


def evaluate_summary(judgement, summary, tokenizer, model, max_tokens):

    prompt = build_prompt(tokenizer, judgement, summary)

    for attempt in range(5):
        try:
            output = generate_response(tokenizer, model, prompt, max_tokens)

            if not output:
                print("Empty response. Retrying...")
                time.sleep(2)
                continue

            scores = extract_scores(output)

            if any(v is None for v in scores.values()):
                print("Invalid format detected. Retrying...")
                time.sleep(2)
                continue

            return output

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

    return "ERROR"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i", required=True,
                        help="Dataset directory")

    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    parser.add_argument("--model", "-m", required=True,
                        help="HuggingFace model name")

    parser.add_argument("--hf_token", "-k", required=True,
                        help="HuggingFace token")

    parser.add_argument("--max_tokens", "-t", type=int, default=512)

    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.hf_token)

    os.makedirs(args.output, exist_ok=True)

    raw_output_dir = os.path.join(args.output, "LLM_Judge_Output")
    os.makedirs(raw_output_dir, exist_ok=True)

    csv_path = os.path.join(args.output, "LLM_Judge_Scores.csv")

    rows = []

    folders = sorted([
        f for f in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, f))
    ])

    for idx, folder in enumerate(folders, 1):

        print(f"Processing {folder} ({idx}/{len(folders)})")

        folder_path = os.path.join(args.input, folder)

        judgement_path = os.path.join(folder_path, "Judgement.txt")
        summary_path = os.path.join(folder_path, "Model_Output.txt")

        if not os.path.exists(judgement_path):
            print("Missing Judgement.txt")
            continue

        if not os.path.exists(summary_path):
            print("Missing Model_Output.txt")
            continue

        with open(judgement_path, encoding="utf-8") as f:
            judgement = f.read()

        with open(summary_path, encoding="utf-8") as f:
            summary = f.read()

        llm_output = evaluate_summary(
            judgement,
            summary,
            tokenizer,
            model,
            args.max_tokens
        )

        txt_path = os.path.join(raw_output_dir, f"{folder}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(llm_output)

        if llm_output == "ERROR":
            print(f"Skipping {folder} due to repeated failure.")
            continue

        scores = extract_scores(llm_output)

        if any(v is None for v in scores.values()):
            print(f"Skipping {folder} due to invalid scores.")
            continue

        rows.append([
            folder,
            scores["RC"],
            scores["FC"],
            scores["LTE"],
            scores["Gr"],
            scores["Co"],
            scores["CH"]
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "ID",
            "Relevance & Coverage",
            "Factual Consistency",
            "Legal Terms Explained",
            "Grammaticality",
            "Coherence",
            "Conciseness & Hallucination"
        ])

        writer.writerows(rows)

    print("\nEvaluation Completed")
    print(f"CSV saved at: {csv_path}")


if __name__ == "__main__":
    main()