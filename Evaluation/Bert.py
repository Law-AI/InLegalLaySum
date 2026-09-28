#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from evaluate import load


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bertscore = load("bertscore")

    ids = sorted([
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ])

    print(f"Total test folders found: {len(ids)}")

    references = []
    predictions = []
    valid_ids = []

    skipped = 0

    for id_ in ids:
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

        references.append(ref_text)
        predictions.append(hyp_text)
        valid_ids.append(id_)

    if len(predictions) == 0:
        raise ValueError("No valid prediction–reference pairs found.")

    # ===== COMPUTE BERTSCORE =====
    scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        device=device,
        rescale_with_baseline=False
    )

    precisions = scores["precision"]
    recalls = scores["recall"]
    f1s = scores["f1"]

    # ===== PER-FILE CSV =====
    per_file_rows = []
    for i, id_ in enumerate(valid_ids):
        per_file_rows.append({
            "id": id_,
            "BERT_P": round(precisions[i] * 100, 4),
            "BERT_R": round(recalls[i] * 100, 4),
            "BERT_F1": round(f1s[i] * 100, 4)
        })

    df = pd.DataFrame(per_file_rows)
    csv_path = os.path.join(output_dir, "BERTScore_PerFile.csv")
    df.to_csv(csv_path, index=False)

    avg_p = round(np.mean(precisions) * 100, 4)
    avg_r = round(np.mean(recalls) * 100, 4)
    avg_f1 = round(np.mean(f1s) * 100, 4)

    txt_path = os.path.join(output_dir, "BERTScore_Average.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BERTScore Evaluation Results\n")
        f.write(f"Used files   : {len(valid_ids)}\n")
        f.write(f"Skipped files: {skipped}\n\n")
        f.write(f"Precision: {avg_p}\n")
        f.write(f"Recall   : {avg_r}\n")
        f.write(f"F1       : {avg_f1}\n")

    print("\n===== AVERAGE BERTSCORE (%) =====")
    print(f"Precision: {avg_p}")
    print(f"Recall   : {avg_r}")
    print(f"F1       : {avg_f1}")

    print("\nSaved files:")
    print(f"- {csv_path}")
    print(f"- {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute BERTScore (English) with Per-file CSV and Average TXT"
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
        help="Directory where BERTScore results will be saved"
    )

    args = parser.parse_args()
    main(args)
