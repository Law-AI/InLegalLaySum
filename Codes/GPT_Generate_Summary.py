#!/usr/bin/env python3

import os
import argparse
import time
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a legal expert skilled in simplifying complex legal summaries.

Rewrite the provided layman summary so it is easy for a high school student
(11th–12th grade level) to understand.

Requirements:
- Use simple vocabulary and short, clear sentences.
- Preserve all original structure, including paragraph breaks and sections if present.
- Keep each paragraph logically consistent and aligned with the original idea.
- Clarify or explain legal terms using plain English only when necessary.
- Ensure the rewritten version is easy to read, with a Flesch-Kincaid Grade Level below 10.

Do NOT:
- Add FKGL scores.
- Add section headers unless already present.
- Use bullet points.
- Add introductions, conclusions, summaries, or personal opinions.

Return only the rewritten version with the original structure preserved.
"""

USER_PROMPT_TEMPLATE = """
Layman Summary:

<SUMMARY>

Rewritten Version:
"""


def simplify_summary(summary_text, model_name, max_tokens):
    """
    Simplifies a single summary using the specified OpenAI model.
    Retries up to 3 times if output is empty or malformed.
    """

    prompt = USER_PROMPT_TEMPLATE.replace(
        "<SUMMARY>",
        summary_text.strip()
    )

    for attempt in range(3):
        try:
            response = client.responses.create(
                model=model_name,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                max_output_tokens=max_tokens,
            )

            output = ""

            if hasattr(response, "output_text") and response.output_text:
                output = response.output_text.strip()

            elif hasattr(response, "output") and response.output:
                if isinstance(response.output, list):
                    first_output = response.output[0]

                    if (
                        first_output
                        and hasattr(first_output, "content")
                        and first_output.content
                    ):
                        first_content = first_output.content[0]

                        if (
                            first_content
                            and hasattr(first_content, "text")
                            and first_content.text
                        ):
                            output = first_content.text.strip()

            if output:
                return output

            print(f"  Retry {attempt + 1}: Empty response.")
            time.sleep(2)

        except Exception as e:
            print(f"  Error during attempt {attempt + 1}: {e}")
            time.sleep(2)

    return "Simplification could not be generated after multiple attempts."


def main():

    parser = argparse.ArgumentParser(
        description="Generate simplified summaries from Groundtruth_Summary.txt files."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Root folder containing case subfolders."
    )

    parser.add_argument(
        "--model",
        "-m",
        default="gpt-5",
        help="Model name (e.g. gpt-5, gpt-4o)."
    )

    parser.add_argument(
        "--max_tokens",
        "-t",
        type=int,
        default=1024,
        help="Maximum output tokens."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Simplified_Summary.txt files."
    )

    args = parser.parse_args()

    folders = sorted([
        f for f in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, f))
    ])

    print(f"Found {len(folders)} folders.\n")

    for idx, folder in enumerate(folders, start=1):

        folder_path = os.path.join(args.input, folder)

        summary_path = os.path.join(
            folder_path,
            "Groundtruth_Summary.txt"
        )

        output_path = os.path.join(
            folder_path,
            "Simplified_Summary.txt"
        )

        print(f"[{idx}/{len(folders)}] Processing {folder}")

        if not os.path.exists(summary_path):
            print("  Skipped - Groundtruth_Summary.txt not found")
            continue

        if os.path.exists(output_path) and not args.overwrite:
            print("  Skipped - Simplified_Summary.txt already exists")
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read().strip()

            if not summary_text:
                print("  Skipped - Groundtruth_Summary.txt is empty")
                continue

            simplified_text = simplify_summary(
                summary_text,
                args.model,
                args.max_tokens
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(simplified_text)

            print(
                f"  Saved -> Simplified_Summary.txt "
                f"({len(simplified_text.split())} words)"
            )

        except Exception as e:
            print(f"  Error: {e}")

    print("\nCompleted.")


if __name__ == "__main__":
    main()