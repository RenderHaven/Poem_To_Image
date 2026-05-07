# %%
from dotenv import load_dotenv
import os
load_dotenv()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# %%
SEMANTIC_EXPERT = """
You are a Semantic Expert specializing in photoreali
stic visual composition. Extract concrete, camera-ready details: subjects/
objects, environmental settings, natural lighting conditions, depth of fie
ld, and compositional elements. Emphasize photographic realism with specif
ic camera angles, lens characteristics, and cinematic framing. Focus on ta
ngible, depictable elements that translate directly to high-quality photog
raphic imagery.
"""

AFFECTIVE_EXPERT = """
You are an Affective Expert focused on atmospheric ph
otography. Describe mood through cinematic lighting, color grading, weathe
r conditions, and environmental textures that create emotional resonance.
Use photographic terminology: golden hour lighting, dramatic shadows, soft
focus, depth of field, and color temperature. Express emotions through vis
ual atmosphere rather than abstract labels, emphasizing photorealistic moo
d creation.
"""

CULTURAL_EXPERT = """
You are a Cultural Expert specializing in documentary
photography. Identify authentic cultural, historical, and geographical ele
ments: period-appropriate architecture, traditional attire, cultural artif
acts, historical settings, and regional characteristics. Focus on photojou
rnalistic authenticity with accurate period details, cultural symbols, and
environmental context. Maintain documentary realism and avoid speculative
elements.
"""

GENRE_EXPERT = """
You are a Genre Expert focused on photographic style a
nd aesthetic periods. Translate literary genres into visual photography st
yles: vintage film aesthetics, documentary realism, fine art photography,
or contemporary digital styles. Consider historical photographic movement
s, color palettes, composition rules, and technical characteristics that d
efine different photographic genres. Emphasize authentic period-appropriat
e visual styles.
"""

GENDER_EXPERT = """
You are a Gender Expert specializing in portrait and ch
aracter photography. Identify visual cues for subjects: age, gender presen
tation, social context, and relational dynamics when textually supported.
Focus on authentic portrait photography elements: natural expressions, app
ropriate attire, social settings, and interpersonal dynamics. Maintain res
pectful, realistic representation without stereotyping.
"""

SYNTHESIZER = """
You are a Professional Prompt Synthesizer for
photorealistic image generation. Fuse expert insights into coherent, camer
a-ready prompts optimized for FLUX models. Emphasize photographic realism,
natural lighting, authentic details, and professional composition. Ensure
prompts guide creation of high-quality, lifelike imagery with proper techn
ical specifications.
"""


# %%
MODEL_PATH = os.getenv("OLMoE_MODEL_PATH","models/OLMoE-1B-7B-0924-Instruct")
DATASET_PATH = os.getenv("POEM_DATASET_PATH","datasets/poemsum_test.csv")

# %%
# ================= LOAD DATASET =================
import pandas as pd
df = pd.read_csv(DATASET_PATH)
df.head()

# %%
# ================= LOAD MODEL =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# %%
# ================= GENERATE FUNCTION =================
def generate(system_prompt, poem):
    prompt = f"""<|system|>
{system_prompt}

<|user|>
Poem:
{poem}

Generate the expert visual description.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("Generate the expert visual description.")[-1].strip()

# %%
def process_poem(id,poem):
    """
    Generate all expert prompts + final synthesized prompt
    for a single poem and return as dict.
    """

    semantic = generate(SEMANTIC_EXPERT, poem)
    affective = generate(AFFECTIVE_EXPERT, poem)
    cultural = generate(CULTURAL_EXPERT, poem)
    genre = generate(GENRE_EXPERT, poem)
    gender = generate(GENDER_EXPERT, poem)

    # -------- Synthesizer --------
    synth_input = f"""
Poem:
{poem}

Semantic Expert:
{semantic}

Affective Expert:
{affective}

Cultural Expert:
{cultural}

Genre Expert:
{genre}

Gender Expert:
{gender}
"""

    final_prompt = generate(SYNTHESIZER, synth_input)

    return {
        "id":id,
        "poem": poem,
        "semantic": semantic,
        "affective": affective,
        "cultural": cultural,
        "genre": genre,
        "gender": gender,
        "final_prompt": final_prompt
    }


# %%
print(df.iloc[0]['ctext'])
ans=generate(CULTURAL_EXPERT,df.iloc[0]['ctext'])
ans

# %%
import os

results = []

SAVE_INTERVAL = 1
OUTPUT_FILE = os.getenv("PROMPT_OUTPUT_CSV","output_datasets/generated_prompts.csv")

# ================= LOAD OLD OUTPUT IF EXISTS =================
start_index = 0

if os.path.exists(OUTPUT_FILE):
    old_df = pd.read_csv(OUTPUT_FILE)

    # keep old results
    results = old_df.to_dict("records")

    # number of poems already processed
    start_index = len(old_df)

    print(f"Resuming from row {start_index}")




# %%
# ================= MAIN LOOP =================
for idx, row in df.iloc[start_index:].iterrows():
    id=row["id"]
    poem = row["ctext"]

    poem_result = process_poem(id,poem)
    results.append(poem_result)

    # ⭐ periodic save
    if (len(results)) % SAVE_INTERVAL == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved progress at {len(results)} poems")

# ================= FINAL SAVE =================
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)
print("Final save completed")


