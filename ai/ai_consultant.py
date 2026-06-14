import ollama
import json

def generate_ai_recommendations(prompt):
    response = ollama.chat(model="llama3",
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.2, "num_predict": 250, "top_k": 20, "top_p": 0.8})

    return response["message"]["content"]


def generate_category_mapping(column_name, unique_values):
    values_list = ", ".join([f'"{v}"' for v in unique_values])

    prompt = f"""You are a data cleaning assistant.

The column "{column_name}" contains these distinct values:
{values_list}

Many of these values likely represent the same underlying category, written differently (different casing, abbreviations, spelling variants, or synonyms).

Return ONLY a JSON object mapping each original value to a single standardized canonical label. The canonical label should be a clean, title-cased, human-readable version. Do not invent new categories that aren't represented. Do not merge values that represent genuinely different categories.

Example format:
{{"f": "Female", "female": "Female", "FEMALE": "Female", "m": "Male", "Male": "Male"}}

Return ONLY the JSON object, no explanation, no markdown formatting.
"""
    
    response = ollama.chat(model="llama3",
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.0, "num_predict": 500, "top_k": 10, "top_p": 0.5})
    raw = response["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            return None
        return {str(k): str(v) for k, v in mapping.items()}
    except (json.JSONDecodeError, ValueError):
        return None