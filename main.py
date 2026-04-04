import os
import re
import base64
import io
import json
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"🔑 KEY BEING USED: {api_key[:20] if api_key else 'NO KEY FOUND'}")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def compress_image(image_bytes: bytes) -> tuple[str, str]:
    """Compress image to under 1MB and return base64 + mime_type"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=60)
    output.seek(0)
    compressed = output.read()
    print(f"📦 Image compressed: {len(image_bytes)//1024}KB → {len(compressed)//1024}KB")
    return base64.b64encode(compressed).decode("utf-8"), "image/jpeg"

def parse_nutrition_response(text: str) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    parsed = {
        "dish_name": None,
        "estimated_portion": None,
        "calories": None,
        "protein_g": None,
        "carbs_g": None,
        "fat_g": None,
    }
    for line in lines:
        lower = line.lower()
        if "dish" in lower and "name" in lower:
            parsed["dish_name"] = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        elif "estimated" in lower and "portion" in lower:
            parsed["estimated_portion"] = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        elif "calorie" in lower:
            m = re.search(r"(\d+[\.]?\d*)", line)
            if m: parsed["calories"] = float(m.group(1))
        elif "protein" in lower:
            m = re.search(r"(\d+[\.]?\d*)", line)
            if m: parsed["protein_g"] = float(m.group(1))
        elif "carb" in lower:
            m = re.search(r"(\d+[\.]?\d*)", line)
            if m: parsed["carbs_g"] = float(m.group(1))
        elif "fat" in lower:
            m = re.search(r"(\d+[\.]?\d*)", line)
            if m: parsed["fat_g"] = float(m.group(1))
    if not parsed["dish_name"] and lines:
        parsed["dish_name"] = lines[0]
    if not parsed["estimated_portion"] and len(lines) > 1:
        parsed["estimated_portion"] = lines[1]
    return parsed


# ── ANALYZE FOOD ENDPOINT ─────────────────────────────────
@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image, mime_type = compress_image(image_bytes)

        prompt = (
            "You are an expert nutritionist with deep knowledge of all global cuisines "
            "(including Western, Asian, Indian, and Mediterranean). Analyze the image carefully "
            "to identify the specific food or dish. If it is a regional specialty, name it accurately. "
            "Provide output in this exact format:\n"
            "Dish Name: <name>\n"
            "Estimated Portion: <portion>\n"
            "Total Calories: <kcal>\n"
            "Protein: <g>\n"
            "Carbs: <g>\n"
            "Fat: <g>\n"
            "Return only plain text with no additional explanation."
        )

        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        text_output = response.choices[0].message.content
        parsed = parse_nutrition_response(text_output)

        return {
            "success": True,
            "raw_response": text_output,
            "parsed": parsed,
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str

@app.post("/search-food")
async def search_food(request: SearchRequest):
    try:
        prompt = f"""You are an expert nutritionist with deep knowledge of all global cuisines, including Indian food.
The user searched for: "{request.query}"

Provide accurate nutrition information for this specific food item.
Respond ONLY in this exact format — no extra text, no markdown, no explanation:
Dish Name: <exact name>
Estimated Portion: <standard portion>
Total Calories: <kcal number only>
Protein: <grams number only>
Carbs: <grams number only>
Fat: <grams number only>"""
        print(f"🔍 Searching AI for: {request.query}")
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}]
        )
        text_output = response.choices[0].message.content.strip()
        parsed = parse_nutrition_response(text_output)
        return {"success": True, "raw_response": text_output, "parsed": parsed}
    except Exception as e:
        print(f"❌ SEARCH ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── PANTRY ARCHITECT ENDPOINT ─────────────────────────────
class PantryRequest(BaseModel):
    ingredients: List[str]
    goal: str

@app.post("/generate-recipes")
async def generate_recipes(request: PantryRequest):
    try:
        ingredients_str = ", ".join(request.ingredients)

        goal_descriptions = {
            "lose":     "weight loss (low calorie, high fibre, high protein, keeps user full)",
            "muscle":   "muscle gain (very high protein, adequate complex carbs, calorie surplus)",
            "maintain": "weight maintenance (balanced macros, variety, moderation)",
            "athletic": "athletic performance (high complex carbs for energy, anti-inflammatory, fast recovery)"
        }
        goal_desc = goal_descriptions.get(request.goal, "general health")

        prompt = f"""You are an expert nutritionist and chef specialising in Indian and global cuisines.

The user has these ingredients available: {ingredients_str}
Their fitness goal is: {goal_desc}

Generate exactly 3 recipes using ONLY these ingredients.
You may suggest maximum 1-2 optional extra ingredients per recipe.
Rank them from best to worst match for their goal.
Make the recipes realistic, practical and goal-aligned.

Respond ONLY with this exact JSON format — no extra text, no markdown, no explanation:
{{
  "recipes": [
    {{
      "rank": 1,
      "emoji": "🍛",
      "name": "Recipe Name",
      "description": "1-2 sentence description of the dish and why it suits the goal",
      "goalMatchPct": 92,
      "goalMatchReason": "Short reason why this matches the goal",
      "cal": 350,
      "protein": 25,
      "carbs": 40,
      "fat": 8,
      "fiber": 6,
      "portion": "1 bowl (300g)",
      "ingredientsUsed": ["ingredient1", "ingredient2", "ingredient3"],
      "optionalBoosts": ["optional ingredient"],
      "cookingSteps": [
        "Step 1 description",
        "Step 2 description",
        "Step 3 description",
        "Step 4 description"
      ],
      "nutritionistTweak": "One specific professional tip to make it even healthier for the goal",
      "satietyScore": 4.5
    }},
    {{
      "rank": 2,
      "emoji": "🥗",
      "name": "Recipe Name",
      "description": "Description",
      "goalMatchPct": 85,
      "goalMatchReason": "Reason",
      "cal": 280,
      "protein": 18,
      "carbs": 35,
      "fat": 6,
      "fiber": 5,
      "portion": "1 plate (250g)",
      "ingredientsUsed": ["ingredient1", "ingredient2"],
      "optionalBoosts": ["optional"],
      "cookingSteps": ["Step 1", "Step 2", "Step 3", "Step 4"],
      "nutritionistTweak": "Professional tip",
      "satietyScore": 3.8
    }},
    {{
      "rank": 3,
      "emoji": "🍲",
      "name": "Recipe Name",
      "description": "Description",
      "goalMatchPct": 78,
      "goalMatchReason": "Reason",
      "cal": 200,
      "protein": 12,
      "carbs": 28,
      "fat": 4,
      "fiber": 4,
      "portion": "1 serving (200g)",
      "ingredientsUsed": ["ingredient1", "ingredient2"],
      "optionalBoosts": ["optional"],
      "cookingSteps": ["Step 1", "Step 2", "Step 3", "Step 4"],
      "nutritionistTweak": "Professional tip",
      "satietyScore": 3.2
    }}
  ]
}}"""

        print(f"🧑‍🍳 Generating recipes for: {ingredients_str} | Goal: {request.goal}")

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text_output = response.choices[0].message.content.strip()
        print(f"🍽️ Raw recipe response: {text_output[:200]}...")

        # Clean JSON response
        if "```json" in text_output:
            text_output = text_output.split("```json")[1].split("```")[0].strip()
        elif "```" in text_output:
            text_output = text_output.split("```")[1].split("```")[0].strip()

        recipes_data = json.loads(text_output)

        print(f"✅ Successfully generated {len(recipes_data['recipes'])} recipes!")

        return {
            "success": True,
            "recipes": recipes_data["recipes"]
        }

    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"❌ Raw output was: {text_output}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try again.")

    except Exception as e:
        print(f"❌ PANTRY ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))