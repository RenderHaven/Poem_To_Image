import os
import io
import csv
import torch
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from diffusers import Flux2KleinPipeline

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

# =====================================================
# CONFIG
# =====================================================
device = "cuda"
dtype = torch.bfloat16

MODEL_PATH = os.getenv(
    "FLUX_MODEL_PATH",
    "./models/FLUX.2-klein-base-9B"
)

CSV_PATH = os.getenv("PROMPT_OUTPUT_CSV")
OUTPUT_CSV = os.getenv(
    "IMAGES_OUTPUT_CSV",
    "output_datasets/generated_images.csv"
)

IMAGE_DIR = os.getenv(
    "IMAGE_DIR",
    "output_images"
)

PROMPT_COLUMNS = [
    "semantic",
    "affective",
    "cultural",
    "genre",
    "gender",
    "final_prompt"
]

EXPECTED_COLUMNS = [
    "id",
    "poem",

    "semantic",
    "semantic_img_path",

    "affective",
    "affective_img_path",

    "cultural",
    "cultural_img_path",

    "genre",
    "genre_img_path",

    "gender",
    "gender_img_path",

    "final_prompt",
    "final_prompt_img_path",
]

# =====================================================
# CREATE DIRS
# =====================================================
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    local_files_only=True
)

pipe = pipe.to(device)

# =====================================================
# IMAGE GENERATION
# =====================================================
def generate_image_bytes(prompt):

    try:

        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.0,
            num_inference_steps=50,
            generator=torch.Generator(
                device=device
            ).manual_seed(0)
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return buffer.getvalue()

    except Exception as e:
        print("Generation error:", e)
        return None

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv(CSV_PATH)

print(df.columns)
print("Rows:", len(df))

# =====================================================
# RESUME LOGIC
# =====================================================
start_index = 0

if os.path.exists(OUTPUT_CSV):

    try:
        existing_df = pd.read_csv(OUTPUT_CSV)

        if len(existing_df) > 0:

            if "id" in existing_df.columns:
                start_index = int(existing_df["id"].max()) + 1
            else:
                start_index = len(existing_df)

        print(f"✅ Resuming from row {start_index}")

    except Exception as e:
        print("⚠ Failed to read existing CSV:", e)
        print("Starting from 0")

write_header = not os.path.exists(OUTPUT_CSV)

# =====================================================
# MAIN LOOP
# =====================================================
for idx, row in tqdm(
    df.iloc[start_index:].iterrows(),
    total=len(df) - start_index
):

    row_dict = {}

    # =================================================
    # BASIC FIELDS
    # =================================================
    row_id = row.get("id", idx)

    row_dict["id"] = row_id
    row_dict["poem"] = str(row.get("poem", ""))

    # =================================================
    # GENERATE FOR EACH PROMPT
    # =================================================
    for col in PROMPT_COLUMNS:

        prompt = str(row.get(col, ""))

        row_dict[col] = prompt

        try:

            img_bytes = generate_image_bytes(prompt)

            img_name = f"{row_id}_{col}.png"

            img_path = os.path.join(
                IMAGE_DIR,
                img_name
            )

            if img_bytes:

                image = Image.open(
                    io.BytesIO(img_bytes)
                )

                image.save(img_path)

                row_dict[f"{col}_img_path"] = img_path

            else:
                row_dict[f"{col}_img_path"] = None

        except Exception as e:

            print(f"❌ Image failed for {col}:", e)

            row_dict[f"{col}_img_path"] = None

    # =================================================
    # SAFE CSV WRITE
    # =================================================
    try:

        row_df = pd.DataFrame([row_dict])

        # enforce consistent column ordering
        row_df = row_df.reindex(
            columns=EXPECTED_COLUMNS
        )

        row_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar="\\"
        )

        write_header = False

    except Exception as e:

        print(f"❌ CSV write failed at row {row_id}:", e)

print("\n✅ Image generation completed.")