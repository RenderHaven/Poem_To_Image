# Poem → Image Generation using Multi-Expert Prompting

This project generates **images from poems** using a **multi-expert
prompt generation system** combined with **diffusion-based image
generation and fusion**.

Instead of relying on a single prompt, the poem is analyzed from
multiple perspectives (semantic, emotional, cultural, stylistic, etc.).
Each perspective generates its own visual interpretation, which is later
**combined into a final image**.

This approach captures **multiple layers of poetic meaning**, producing
images that better represent the poem.

------------------------------------------------------------------------

# Project Structure

    project-root
    │
    ├── datasets/                     # Input datasets (PoemSum dataset, etc.)
    │
    ├── models/                       # Downloaded model weights
    │   ├── OLMoE-1B-7B-0924-Instruct
    │   └── FLUX.2-klein-base-9B
    │
    ├── output/                       # Generated outputs
    │   ├── generated_prompts.csv
    │   ├── generated_images.csv
    │   └── final_images.csv
    │
    ├── output_images/                # All generated images
    │
    ├── scripts/
    │   ├── generate_prompts.py       # Step 1 – Multi-expert prompt generation
    │   ├── generate_images.py        # Step 2 – Image generation from prompts
    │   └── generate_final_images.py  # Step 3 – Image fusion (T1, T2, T3)
    │
    ├── setup.sh                      # Setup script for environment
    ├── requirements.txt              # Python dependencies
    └── README.md

------------------------------------------------------------------------

# Setup

## 1. Clone the repository

``` bash
git clone https://github.com/your-repo/poem-to-image.git
cd poem-to-image
```

## 2. Run the setup script

``` bash
bash setup.sh
```

This script installs required dependencies and prepares the environment.

## 3. Install dependencies manually (optional)

``` bash
pip install -r requirements.txt
```

## 4. Download models

Place the required models inside the `models/` directory.

Example:

    models/
    │
    ├── OLMoE-1B-7B-0924-Instruct
    └── FLUX.2-klein-base-9B

------------------------------------------------------------------------

# What We Did

Our goal is to generate an image that **represents a poem more
accurately**.

Poems often contain multiple layers:

-   literal imagery
-   emotions
-   cultural references
-   artistic style

Traditional text-to-image systems rely on **one prompt**, which often
fails to capture all of these layers.

Our system analyzes the poem using **multiple expert perspectives**.

## Expert Roles

### Semantic Expert

Extracts literal visual elements:

-   subjects
-   objects
-   environments
-   actions

### Emotion (Affective) Expert

Transforms emotions into visual atmosphere.

Example:

    "sad" → "dim evening light with soft shadows and cool colors"

### Cultural Expert

Identifies contextual references:

-   historical context
-   cultural symbols
-   geographical elements

### Genre Expert

Recognizes literary style such as:

-   elegy
-   romantic poetry
-   surreal imagery

These cues influence the **visual aesthetic**.

### Gender Expert

Infers gender cues only when supported by the text.

Defaults to **unspecified** when unclear to ensure respectful
representation.

Each expert generates **its own prompt and image**.

These images are then **merged** to produce a final image representing
the poem more faithfully.

------------------------------------------------------------------------

# Pipeline Overview

    Poem
      ↓
    Multi-Expert Prompt Generation
      ↓
    Semantic / Emotion / Cultural / Genre / Gender Prompts
      ↓
    Image Generation (FLUX)
      ↓
    Expert Images
      ↓
    Image Fusion
      ↓
    Final Images (T1, T2, T3)

------------------------------------------------------------------------

# Models Used

## OLMoE-1B-7B-Instruct

Instruction-tuned LLM used for **multi-expert prompt generation**.

Roles performed by this model:

-   Semantic expert
-   Emotion expert
-   Cultural expert
-   Genre expert
-   Gender expert
-   Prompt synthesizer

------------------------------------------------------------------------

## FLUX / FLUX.1 Kontext

Diffusion-based image generation model used for:

-   generating expert images
-   merging images
-   final image synthesis

Model link:

https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

------------------------------------------------------------------------

## CLIP

Used for **text--image alignment evaluation**.

Measures how well generated images match the prompts.

------------------------------------------------------------------------

## Long-CLIP

Extension of CLIP designed for **long textual inputs such as poems**.

Used for evaluating **poem-to-image alignment**.

------------------------------------------------------------------------

# Frameworks & Techniques

## Mixture of Experts (MoE)

Multiple expert prompts are generated for the same poem.

Each expert focuses on **a different semantic dimension**.

------------------------------------------------------------------------

## Retrieval-Augmented Generation (RAG)

Used to ground expert outputs using **external literary context**.

------------------------------------------------------------------------

## Zero-Shot Prompting

Experts operate using **instruction prompts without model fine-tuning**.

------------------------------------------------------------------------

## Image Fusion

Expert-generated images are merged using **FLUX-based diffusion
fusion**.

------------------------------------------------------------------------

# Dataset

## PoemSum Dataset

Used for poem inputs and evaluation samples.

Repository:

https://github.com/Ridwan230/PoemSum

------------------------------------------------------------------------

# Evaluation Methods

## CLIP Score

Measures **text--image alignment**.

Higher score indicates better semantic correspondence.

------------------------------------------------------------------------

## Long-CLIP Score

Measures **alignment between long poems and generated images**.

------------------------------------------------------------------------

## Human Evaluation

Human judges evaluate images using a **5-point Likert scale**.

  Score   Meaning
  ------- -----------
  1       Poor
  2       Weak
  3       Average
  4       Good
  5       Excellent

Evaluation criteria:

-   relevance to poem
-   visual quality
-   emotional representation
-   cultural authenticity

------------------------------------------------------------------------

# Running the Pipeline

## Step 1 --- Generate Prompts

``` bash
python scripts/generate_prompts.py
```

Output:

    generated_prompts.csv

------------------------------------------------------------------------

## Step 2 --- Generate Expert Images

``` bash
python scripts/generate_images.py
```

Outputs:

-   expert images
-   `generated_images.csv`

------------------------------------------------------------------------

## Step 3 --- Generate Final Images

``` bash
python scripts/generate_final_images.py
```

Outputs:

-   **T1 images**
-   **T2 merged images**
-   **T3 regenerated images**

------------------------------------------------------------------------

# Example Output

For each poem:

    semantic_image.png
    affective_image.png
    cultural_image.png
    genre_image.png
    gender_image.png

    T1.png
    T2.png
    T3.png

------------------------------------------------------------------------

# Key Idea

Traditional systems:

    Poem → Prompt → Image

Our approach:

    Poem
     ↓
    Multi-Expert Analysis
     ↓
    Multiple Visual Interpretations
     ↓
    Image Fusion
     ↓
    Final Image

This allows the system to capture **multiple layers of poetic meaning**.

------------------------------------------------------------------------

# Future Improvements

Possible future work:

-   CLIP-guided prompt refinement
-   layout-aware diffusion
-   multimodal reasoning models
-   improved poem semantic parsing