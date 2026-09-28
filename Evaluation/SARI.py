#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluate import load


def clean_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("\xad", "")
    return text


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


def main(args):
    test_root = args.test_dataset
    gen_root = args.generated_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    sari = load("sari")

    ids = sorted([
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ])

    print(f"Total test folders found: {len(ids)}")

    per_file_results = []
    sari_values = []

    skipped = 0

    for id_ in tqdm(ids, desc="Computing SARI", unit="file"):
        src_path = os.path.join(test_root, id_, "Judgement.txt")
        ref_path = os.path.join(test_root, id_, "Simplified_Summary.txt")
        hyp_path = os.path.join(gen_root, f"{id_}.txt")

        if not (os.path.exists(src_path) and os.path.exists(ref_path) and os.path.exists(hyp_path)):
            skipped += 1
            continue

        source = read_file(src_path)
        reference = read_file(ref_path)
        prediction = read_file(hyp_path)

        if len(source) < 10 or len(reference) < 10 or len(prediction) < 10:
            skipped += 1
            continue

        score = sari.compute(
            sources=[source],
            predictions=[prediction],
            references=[[reference]]
        )["sari"]

        sari_values.append(score)

        per_file_results.append({
            "id": id_,
            "SARI": round(score, 4)
        })

    if not sari_values:
        raise ValueError("No valid files found for SARI computation.")

    csv_path = os.path.join(output_dir, "SARI_PerFile.csv")
    pd.DataFrame(per_file_results).to_csv(csv_path, index=False)

    avg_sari = round(float(np.mean(sari_values)), 4)

    txt_path = os.path.join(output_dir, "SARI_Average.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SARI Evaluation Results\n")
        f.write("=======================\n")
        f.write(f"Total files used : {len(sari_values)}\n")
        f.write(f"Skipped files   : {skipped}\n\n")
        f.write(f"Average SARI    : {avg_sari}\n")

    print("\n===== SARI EVALUATION COMPLETE =====")
    print(f"Average SARI : {avg_sari}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved TXT    : {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SARI evaluation with per-file CSV and average TXT"
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
        help="Path to model output folder (e.g., Output_Llama)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save SARI results"
    )

    args = parser.parse_args()
    main(args)
