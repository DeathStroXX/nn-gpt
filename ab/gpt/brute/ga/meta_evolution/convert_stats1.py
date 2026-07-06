#!/usr/bin/env python3
"""
convert_stats1.py
-----------------
Iterates over all directories inside stats1/ and re-structures 1.json files.
Extracts the "hyperparameters" block, wraps it inside a list [hyperparameters],
and overwrites 1.json with this new format.
"""

import os
import json

# BASE_DIR = "/shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/stats1/img-classification_cifar-10_GenFractalNet-0a41a6790da954dbb02b0349084634af"
# BASE_DIR = "/shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/stats1"
BASE_DIR = "/shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/baseline_stats1"

def main():
    if not os.path.isdir(BASE_DIR):
        print(f"Error: BASE_DIR does not exist: {BASE_DIR}")
        return

    updated = 0
    failed = 0
    skipped = 0

    # print(f"Scanning stats directories inside: {BASE_DIR}\n")
    # 
    # for folder_name in sorted(os.listdir(BASE_DIR)):
    #     folder_path = os.path.join(BASE_DIR, folder_name)
    # 
    #     # Skip non-directories
    #     if not os.path.isdir(folder_path):
    #         continue

    # Check if BASE_DIR itself is a model folder containing 1.json directly
    direct_json = os.path.join(BASE_DIR, "1.json")
    if os.path.isfile(direct_json):
        print(f"Targeting single model folder: {BASE_DIR}\n")
        targets = [BASE_DIR]
    else:
        print(f"Scanning stats directories inside: {BASE_DIR}\n")
        targets = [os.path.join(BASE_DIR, name) for name in sorted(os.listdir(BASE_DIR))]

    for folder_path in targets:
        # Skip non-directories
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "1.json")

        # Skip if 1.json doesn't exist
        if not os.path.exists(json_path):
            print(f"Missing file: {json_path}")
            failed += 1
            continue

        try:
            # Load original JSON
            with open(json_path, "r") as f:
                data = json.load(f)

            # Safety check: if it is already a list, skip it
            if isinstance(data, list):
                print(f"Skipped (already a list): {json_path}")
                skipped += 1
                continue

            # Extract hyperparameters
            hyperparameters = data.get("hyperparameters", {})

            # Wrap inside list
            new_data = [hyperparameters]

            # Overwrite file
            with open(json_path, "w") as f:
                json.dump(new_data, f, indent=4)

            print(f"Updated: {json_path}")
            updated += 1

        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            failed += 1

    print("\nFinished")
    print(f"Updated files: {updated}")
    print(f"Skipped files: {skipped}")
    print(f"Failed files:  {failed}")

if __name__ == "__main__":
    main()
