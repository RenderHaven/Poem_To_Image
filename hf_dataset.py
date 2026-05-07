import math
import os
import pandas as pd

from dotenv import load_dotenv

from datasets import (
    Dataset,
    Features,
    Value,
    Image
)

load_dotenv()

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = os.getenv("FINAL_CSV")
OUTPUT_DIR = os.getenv("HF_DATASET_DIR")

# increased chunk size
CHUNK_SIZE = 120

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD CSV
# =====================================================
df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))
print(df.columns.tolist())

# =====================================================
# IMAGE MAPPING
# =====================================================
IMAGE_MAPPING = {
    "semantic_image": "semantic_img_path",
    "affective_image": "affective_img_path",
    "cultural_image": "cultural_img_path",
    "genre_image": "genre_img_path",
    "gender_image": "gender_img_path",
    "T1_image": "T1_img_path",
    "T2_image": "T2_img_path",
    "T3_image": "T3_img_path",
}

# =====================================================
# HF FEATURES
# =====================================================
features = Features({

    "id": Value("int64"),
    "poem": Value("string"),

    "semantic": Value("string"),
    "affective": Value("string"),
    "cultural": Value("string"),
    "genre": Value("string"),
    "gender": Value("string"),
    "final_prompt": Value("string"),

    "semantic_image": Image(decode=True),
    "affective_image": Image(decode=True),
    "cultural_image": Image(decode=True),
    "genre_image": Image(decode=True),
    "gender_image": Image(decode=True),

    "T1_image": Image(decode=True),
    "T2_image": Image(decode=True),
    "T3_image": Image(decode=True),
})

# =====================================================
# CREATE CHUNKS
# =====================================================
num_chunks = math.ceil(len(df) / CHUNK_SIZE)

print("Chunks:", num_chunks)

for i in range(num_chunks):

    start = i * CHUNK_SIZE
    end = min((i + 1) * CHUNK_SIZE, len(df))

    chunk_df = df.iloc[start:end]

    records = []

    # -------------------------------------------------
    # ROW BY ROW
    # -------------------------------------------------
    for _, row in chunk_df.iterrows():

        item = {}

        # ---------------------------------------------
        # TEXT FIELDS
        # ---------------------------------------------
        item["id"] = int(row["id"])

        item["poem"] = str(row["poem"])

        item["semantic"] = str(row["semantic"])
        item["affective"] = str(row["affective"])
        item["cultural"] = str(row["cultural"])
        item["genre"] = str(row["genre"])
        item["gender"] = str(row["gender"])

        item["final_prompt"] = str(
            row["final_prompt"]
        )

        # ---------------------------------------------
        # EMBED IMAGE BYTES
        # ---------------------------------------------
        for out_col, csv_col in IMAGE_MAPPING.items():

            value = row[csv_col]

            if pd.isna(value):

                item[out_col] = None
                continue

            path = str(value)

            if not os.path.exists(path):

                item[out_col] = None
                continue

            # IMPORTANT:
            # read actual image bytes
            with open(path, "rb") as f:

                item[out_col] = {
                    "bytes": f.read(),
                    "path": os.path.basename(path)
                }

        records.append(item)

    # -------------------------------------------------
    # CREATE HF DATASET
    # -------------------------------------------------
    dataset = Dataset.from_list(
        records,
        features=features
    )

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    output_path = os.path.join(
        OUTPUT_DIR,
        f"train-{i:05d}-of-{num_chunks:05d}.parquet"
    )

    dataset.to_parquet(output_path)

    print(
        "Saved:",
        output_path,
        "| Size:",
        round(
            os.path.getsize(output_path) / (1024 * 1024),
            2
        ),
        "MB"
    )

print("\n✅ HF dataset parquet chunks created.")