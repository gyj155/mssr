"""
Download and prepare datasets for MSSR evaluation.

Usage:
    python scripts/prepare_datasets.py --dataset all          # Both datasets
    python scripts/prepare_datasets.py --dataset mmsi         # MMSI-Bench only
    python scripts/prepare_datasets.py --dataset vsbench      # ViewSpatial-Bench only
"""

import argparse
import json
import os
import sys
import zipfile


def prepare_mmsi(output_dir):
    """Download and prepare MMSI-Bench dataset."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    mmsi_dir = os.path.join(output_dir, "MMSI-Bench")
    os.makedirs(mmsi_dir, exist_ok=True)

    json_path = os.path.join(mmsi_dir, "mmsi_bench.json")
    images_dir = os.path.join(mmsi_dir, "images")

    if os.path.exists(json_path) and os.path.exists(images_dir):
        print(f"[MMSI] Already prepared at {mmsi_dir}, skipping.")
        return

    # Step 1: Download parquet
    parquet_path = os.path.join(mmsi_dir, "MMSI_Bench.parquet")
    if not os.path.exists(parquet_path):
        print("[MMSI] Downloading MMSI_Bench.parquet from HuggingFace...")
        hf_hub_download(
            "RunsenXu/MMSI-Bench",
            "MMSI_Bench.parquet",
            repo_type="dataset",
            local_dir=mmsi_dir,
        )
    else:
        print("[MMSI] Parquet already exists, skipping download.")

    # Step 2: Extract images and build JSON
    print("[MMSI] Extracting images and building annotations...")
    df = pd.read_parquet(parquet_path)
    os.makedirs(images_dir, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        qid = int(row["id"])
        image_paths = []
        for img_idx, img_bytes in enumerate(row["images"]):
            fname = f"{qid}_{img_idx}.jpg"
            img_path = os.path.join(images_dir, fname)
            if not os.path.exists(img_path):
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
            image_paths.append(f"./images/{fname}")

        records.append({
            "id": qid,
            "image_paths": image_paths,
            "question_type": row["question_type"],
            "question": row["question"],
            "answer": row["answer"],
            "thought": row.get("thought", ""),
            "image_count": len(image_paths),
        })

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[MMSI] Done: {len(records)} questions, images in {images_dir}")
    print(f"[MMSI] Annotations: {json_path}")


def prepare_vsbench(output_dir):
    """Download and prepare ViewSpatial-Bench dataset."""
    from huggingface_hub import hf_hub_download

    vs_dir = os.path.join(output_dir, "ViewSpatial-Bench")
    os.makedirs(vs_dir, exist_ok=True)

    processed_path = os.path.join(vs_dir, "ViewSpatial-Bench_processed.json")
    if os.path.exists(processed_path):
        print(f"[VS-Bench] Already prepared at {vs_dir}, skipping.")
        return

    # Step 1: Download files
    files_to_download = ["ViewSpatial-Bench.json", "scannetv2_val.zip", "val2017.zip"]
    for fname in files_to_download:
        fpath = os.path.join(vs_dir, fname)
        if not os.path.exists(fpath):
            print(f"[VS-Bench] Downloading {fname}...")
            hf_hub_download(
                "lidingm/ViewSpatial-Bench",
                fname,
                repo_type="dataset",
                local_dir=vs_dir,
            )
        else:
            print(f"[VS-Bench] {fname} already exists, skipping download.")

    # Step 2: Extract zips
    for zip_name in ["scannetv2_val.zip", "val2017.zip"]:
        zip_path = os.path.join(vs_dir, zip_name)
        extract_dir = os.path.join(vs_dir, zip_name.replace(".zip", ""))
        if not os.path.exists(extract_dir):
            print(f"[VS-Bench] Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(vs_dir)
        else:
            print(f"[VS-Bench] {zip_name} already extracted, skipping.")

    # Step 3: Preprocess annotations
    print("[VS-Bench] Preprocessing annotations...")
    raw_path = os.path.join(vs_dir, "ViewSpatial-Bench.json")
    with open(raw_path, "r") as f:
        data = json.load(f)

    processed = []
    for idx, item in enumerate(data):
        # Strip "ViewSpatial-Bench/" prefix, keep only the first image
        # (Scene Simulation questions have multiple images but we use only the first)
        relative = item["image_path"][0].replace("ViewSpatial-Bench/", "./")
        image_paths = [relative]

        # Merge question + choices
        question_with_options = item["question"] + "\nOptions: " + item["choices"].replace("\n", ", ")

        # Extract answer letter (from "A. right" -> "A")
        answer_letter = item["answer"].split(".")[0].strip()

        processed.append({
            "id": idx,
            "image_paths": image_paths,
            "question_type": item["question_type"],
            "question": question_with_options,
            "answer": answer_letter,
            "image_count": len(image_paths),
        })

    with open(processed_path, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"[VS-Bench] Done: {len(processed)} questions")
    print(f"[VS-Bench] Annotations: {processed_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare MSSR datasets")
    parser.add_argument(
        "--dataset",
        choices=["mmsi", "vsbench", "all"],
        default="all",
        help="Which dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output directory (default: dataset)",
    )
    args = parser.parse_args()

    if args.dataset in ("mmsi", "all"):
        prepare_mmsi(args.output_dir)
    if args.dataset in ("vsbench", "all"):
        prepare_vsbench(args.output_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
