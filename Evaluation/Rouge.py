#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from rouge_score import rouge_scorer


def space_handler(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace("\xad", "")
    return text


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return space_handler(f.read())


def main(args):
    test_root = args.test_dataset
    gen_root = args.generated_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    avg_scores = defaultdict(lambda: {'p': 0.0, 'r': 0.0, 'f': 0.0})
    per_file_rows = []

    used = 0
    skipped = 0

    ids = sorted([
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ])

    print(f"Total test folders found: {len(ids)}")

    for id_ in tqdm(ids, desc="Computing ROUGE", unit="file"):
        ref_path = os.path.join(test_root, id_, "Simplified_Summary.txt")
        hyp_path = os.path.join(gen_root, f"{id_}.txt")

        if not os.path.exists(ref_path) or not os.path.exists(hyp_path):
            skipped += 1
            continue

        ref_text = read_file(ref_path)
        hyp_text = read_file(hyp_path)

        if len(ref_text) < 10 or len(hyp_text) < 10:
            skipped += 1
            continue

        scores = scorer.score(ref_text, hyp_text)

        row = {"id": id_}

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            p = scores[metric].precision
            r = scores[metric].recall
            f = scores[metric].fmeasure

            avg_scores[metric]['p'] += p
            avg_scores[metric]['r'] += r
            avg_scores[metric]['f'] += f

            row[f"{metric}_P"] = round(p * 100, 4)
            row[f"{metric}_R"] = round(r * 100, 4)
            row[f"{metric}_F1"] = round(f * 100, 4)

        per_file_rows.append(row)
        used += 1

    if used == 0:
        raise ValueError("No valid file pairs found.")

    for metric in avg_scores:
        for m in avg_scores[metric]:
            avg_scores[metric][m] = round((avg_scores[metric][m] / used) * 100, 4)

    csv_path = os.path.join(output_dir, "ROUGE_PerFile.csv")
    df = pd.DataFrame(per_file_rows)
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_dir, "ROUGE_Scores.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ROUGE Evaluation Results\n")
        f.write(f"Used files   : {used}\n")
        f.write(f"Skipped files: {skipped}\n\n")

        for metric, vals in avg_scores.items():
            f.write(
                f"{metric.upper()} "
                f"P: {vals['p']} | R: {vals['r']} | F1: {vals['f']}\n"
            )

    print("\n===== AVERAGE ROUGE SCORES (%) =====")
    for metric, vals in avg_scores.items():
        print(f"{metric.upper()} -> P: {vals['p']} | R: {vals['r']} | F1: {vals['f']}")

    print("\nSaved files:")
    print(f"- {txt_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-1, ROUGE-2, ROUGE-L with Per-file CSV"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to Test_Dataset folder"
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Path to Output_Llama folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where ROUGE results will be saved"
    )

    args = parser.parse_args()
    main(args)
