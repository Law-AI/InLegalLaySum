#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import pandas as pd
import textstat
from tqdm import tqdm


def space_handler(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace("\xad", "")
    return text


def compute_readability(text):
    gfi = textstat.gunning_fog(text)
    fkgl = textstat.flesch_kincaid_grade(text)
    fre = textstat.flesch_reading_ease(text)
    dale_chall = textstat.dale_chall_readability_score(text)
    coleman_liau = textstat.coleman_liau_index(text)

    return gfi, fkgl, fre, dale_chall, coleman_liau


def main(args):
    folder_path = args.data_dir

    if not os.path.isdir(folder_path):
        raise ValueError("Provided path is not a valid directory")

    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    print(f"Total .txt files found: {len(txt_files)}")

    results = []

    for file_name in tqdm(txt_files, desc="Processing files", unit="file"):
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            text = space_handler(f.read())

        if len(text.strip()) == 0:
            print(f"Skipping empty file: {file_name}")
            continue

        gfi, fkgl, fre, dale_chall, coleman_liau = compute_readability(text)

        results.append({
            "file_name": file_name,
            "GFI": gfi,
            "FKGL": fkgl,
            "FRE": fre,
            "Dale_Chall": dale_chall,
            "Coleman_Liau": coleman_liau
        })


    df = pd.DataFrame(results)

    mean_gfi = df["GFI"].mean()
    mean_fkgl = df["FKGL"].mean()
    mean_fre = df["FRE"].mean()
    mean_dale = df["Dale_Chall"].mean()
    mean_coleman = df["Coleman_Liau"].mean()

    csv_output_path = f"{args.output_prefix}_readability_scores.csv"
    df.to_csv(csv_output_path, index=False)

    txt_output_path = f"{args.output_prefix}_average_readability_scores.txt"
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(f"Average Gunning Fog Index (GFI): {mean_gfi:.4f}\n")
        f.write(f"Average Flesch-Kincaid Grade Level (FKGL): {mean_fkgl:.4f}\n")
        f.write(f"Average Flesch Reading Ease (FRE): {mean_fre:.4f}\n")
        f.write(f"Average Dale-Chall Score: {mean_dale:.4f}\n")
        f.write(f"Average Coleman-Liau Index: {mean_coleman:.4f}\n")

    print("\n===== Readability Statistics =====")
    print(f"Average GFI           : {mean_gfi:.4f}")
    print(f"Average FKGL          : {mean_fkgl:.4f}")
    print(f"Average FRE           : {mean_fre:.4f}")
    print(f"Average Dale-Chall    : {mean_dale:.4f}")
    print(f"Average Coleman-Liau  : {mean_coleman:.4f}")

    print("\nFiles saved:")
    print(f"- {csv_output_path}")
    print(f"- {txt_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Readability Scores (GFI, FKGL, FRE, DC, CLI)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder containing .txt files"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="output",
        help="Prefix name for output files"
    )

    args = parser.parse_args()
    main(args)