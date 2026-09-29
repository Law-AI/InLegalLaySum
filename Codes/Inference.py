import os
import re
import torch
import argparse
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

warnings.filterwarnings("ignore", message="Asking to truncate")

SYSTEM_PROMPT = (
    "You are an expert in Indian law and simplifying legal content. Your task is to summarize complex Indian legal documents so that a reader with up to a high school education can easily understand the summary. Avoid complex legal jargon and explain legal terms in simple words when required. Keep sentences short and clear. Respond only in English."
)

USER_PROMPT_TEMPLATE = (
    "Please write a simplified summary for the following Indian legal judgment so that the summary is suitable for readers up to high school (11th–12th grade). Ensure the summary retains key facts, legal issues, arguments, and the final court decision. Do not add new information or personal opinions.\n\n ### Judgment:\n<JUDGMENT>\n\n### Simplified Summary:"
)

def load_sft_lora_model(base_model_id, adapter_path, load_in_8bit, device):

    print(f"\nLoading tokenizer from base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
        #load_in_8bit=load_in_8bit
    )

    print(f"Loading SFT LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model

def sentence_split(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text, tokenizer, max_len=3500):
    sentences = sentence_split(text)
    chunks, current, token_count = [], [], 0

    for s in sentences:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if token_count + len(ids) > max_len:
            chunks.append(" ".join(current))
            current, token_count = [], 0
        current.append(s)
        token_count += len(ids)

    if current:
        chunks.append(" ".join(current))

    return chunks

def build_prompt(judgment_text):
    user_prompt = USER_PROMPT_TEMPLATE.replace("<JUDGMENT>", judgment_text)
    return f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}"

@torch.no_grad()
def generate_chunk(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3
    )

    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    text = text.strip()

    if text and text[-1] not in ".!?":
        text += "."

    return text

def process(args, model, tokenizer):
    os.makedirs(args.output_folder, exist_ok=True)

    folders = sorted([f.path for f in os.scandir(args.input_folder) if f.is_dir()])

    for idx, folder in enumerate(folders, 1):
        folder_name = os.path.basename(folder)
        input_file = os.path.join(folder, "Judgement.txt")
        output_file = os.path.join(args.output_folder, f"{folder_name}.txt")

        if not os.path.exists(input_file):
            print(f"Skipping {folder_name} — Judgement.txt not found.")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"Skipping {folder_name} — Empty file.")
            continue

        chunks = chunk_text(text, tokenizer)
        print(f"\n[{idx}] Processing {folder_name} - {len(chunks)} chunks")

        per_chunk_tokens = max(1, args.max_new_tokens // len(chunks))
        summaries = []

        for i, chunk in enumerate(chunks, 1):
            print(f"   Generating chunk {i}/{len(chunks)}")
            prompt = build_prompt(chunk)
            summary = generate_chunk(model, tokenizer, prompt, per_chunk_tokens)
            summaries.append(summary)

        final_summary = "\n\n".join(summaries)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_summary)

        print(f" Saved summary - {output_file}")

    print("\nALL FILES PROCESSED SUCCESSFULLY.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--base_model_id", required=True,
                        help="HF ID or local path of base LLM")
    parser.add_argument("--adapter_path", required=True,
                        help="Path to SFT LoRA adapter folder")
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()

    device = "cpu" if args.force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer, model = load_sft_lora_model(
        args.base_model_id,
        args.adapter_path,
        args.load_in_8bit,
        device
    )

    process(args, model, tokenizer)

if __name__ == "__main__":
    main()
