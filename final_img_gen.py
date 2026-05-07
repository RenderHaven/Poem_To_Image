from dotenv import load_dotenv
import os
import csv
import pandas as pd

from tqdm import tqdm

from flux2merger import Flux2Merger

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = os.getenv("IMAGES_OUTPUT_CSV")
OUTPUT_CSV = os.getenv("FINAL_CSV")
IMAGE_DIR = os.getenv("FINAL_IMAGE_DIR")

print(INPUT_CSV, IMAGE_DIR, OUTPUT_CSV)

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# =====================================================
# IMAGE COLUMNS
# =====================================================
IMAGE_COLUMNS = [
    "semantic_img_path",
    "affective_img_path",
    "cultural_img_path",
    "genre_img_path",
    "gender_img_path",
]

# =====================================================
# FINAL COLUMN ORDER
# =====================================================
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

    "T1_img_path",
    "T2_img_path",
    "T3_img_path",
]

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))

# =====================================================
# RESUME LOGIC
# =====================================================
start_index = 0
write_header = not os.path.exists(OUTPUT_CSV)

if os.path.exists(OUTPUT_CSV):

    try:

        existing_df = pd.read_csv(OUTPUT_CSV)

        if len(existing_df) > 0:

            if "id" in existing_df.columns:
                start_index = int(existing_df["id"].max()) + 1
            else:
                start_index = len(existing_df)

            write_header = False

        print(f"✅ Resume from row {start_index}")

    except Exception as e:

        print("⚠ Failed to read existing CSV:", e)
        print("Starting from row 0")

# =====================================================
# LOAD MODEL
# =====================================================
merger = Flux2Merger(
    model_path="./models/FLUX.2-klein-base-9B",

    # better balanced than pentagon
    layout="auto_grid",

    steps=30,
    guidance_scale=2.0,
    seed=0
)

# =====================================================
# MAIN LOOP
# =====================================================
for idx, row in tqdm(
    df.iloc[start_index:].iterrows(),
    total=len(df) - start_index
):

    row_dict = row.to_dict()

    row_id = row.get("id", idx)

    # =================================================
    # VALID IMAGE PATHS
    # =================================================
    image_paths = [
        str(row.get(col))
        for col in IMAGE_COLUMNS
        if pd.notna(row.get(col))
        and os.path.exists(str(row.get(col)))
    ]

    # =================================================
    # T1 → Existing Final Prompt Image
    # =================================================
    row_dict["T1_img_path"] = row.get(
        "final_prompt_img_path",
        None
    )

    # =================================================
    # T2 → Images Only Fusion
    # =================================================
    try:

        if len(image_paths) > 0:

            t2_path = os.path.join(
                IMAGE_DIR,
                f"{row_id}_T2.png"
            )

            merger.generate(
                image_paths=image_paths,

                # IMPORTANT
                prompt="",

                out_path=t2_path
            )

            row_dict["T2_img_path"] = t2_path

            # print(f"✅ T2 Complete → {row_id}")

        else:
            row_dict["T2_img_path"] = None

    except Exception as e:

        print(f"[T2 ERROR] row {row_id}:", e)

        row_dict["T2_img_path"] = None

    # =================================================
    # T3 → Images + Final Prompt
    # =================================================
    try:

        final_prompt = str(
            row.get("final_prompt", "")
        ).strip()

        if final_prompt and len(image_paths) > 0:

            t3_path = os.path.join(
                IMAGE_DIR,
                f"{row_id}_T3.png"
            )

            merger.generate(
                image_paths=image_paths,
                prompt=final_prompt,
                out_path=t3_path
            )

            row_dict["T3_img_path"] = t3_path

            # print(f"✅ T3 Complete → {row_id}")

        else:
            row_dict["T3_img_path"] = None

    except Exception as e:

        print(f"[T3 ERROR] row {row_id}:", e)

        row_dict["T3_img_path"] = None

    # =================================================
    # SAFE CSV SAVE
    # =================================================
    try:

        row_df = pd.DataFrame([row_dict])

        # enforce fixed column order
        row_df = row_df.reindex(
            columns=EXPECTED_COLUMNS
        )

        row_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False,

            # VERY IMPORTANT
            quoting=csv.QUOTE_ALL,
            escapechar="\\"
        )

        write_header = False

    except Exception as e:

        print(f"[CSV ERROR] row {row_id}:", e)

print("\n✅ Final image generation completed.")