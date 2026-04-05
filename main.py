import os
import re
import base64
import io
import json
import asyncio
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
    max_retries=2,  # This natively handles all the retries for you!
    timeout=30.0
)

MODEL_NAME = "openrouter/auto"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit

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
            m = re.search(r"(\d+[.,]?\d*)", line)
            if m: 
                val = float(m.group(1).replace(",", ""))
                if 0 < val <= 5000:
                    parsed["calories"] = val
        elif "protein" in lower:
            m = re.search(r"(\d+[.,]?\d*)", line)
            if m: 
                val = float(m.group(1).replace(",", ""))
                if 0 <= val <= 200:
                    parsed["protein_g"] = val
        elif "carb" in lower:
            m = re.search(r"(\d+[.,]?\d*)", line)
            if m: 
                val = float(m.group(1).replace(",", ""))
                if 0 <= val <= 500:
                    parsed["carbs_g"] = val
        elif "fat" in lower:
            m = re.search(r"(\d+[.,]?\d*)", line)
            if m: 
                val = float(m.group(1).replace(",", ""))
                if 0 <= val <= 200:
                    parsed["fat_g"] = val
    if not parsed["dish_name"] and lines:
        parsed["dish_name"] = lines[0]
    if not parsed["estimated_portion"] and len(lines) > 1:
        parsed["estimated_portion"] = lines[1]
    return parsed


# ── ANALYZE FOOD ENDPOINT ─────────────────────────────────
async def _analyze_food_with_retry(base64_image: str, mime_type: str, max_retries: int = 2):
    """Helper function to analyze food with retry logic"""
    for attempt in range(max_retries + 1):
        try:
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
            print(f"🖼️ Analyzing image (attempt {attempt + 1}/{max_retries + 1})")

            response = client.chat.completions.create(
                model=MODEL_NAME,
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
            
            # Validate parsed output
            if not parsed.get("dish_name"):
                if attempt < max_retries:
                    print(f"⚠️ Retry: Could not parse dish name, retrying...")
                    await asyncio.sleep(1)
                    continue
                raise ValueError("Could not parse food from image.")
            
            required_fields = ["calories", "protein_g", "carbs_g", "fat_g"]
            if any(parsed.get(field) is None for field in required_fields):
                if attempt < max_retries:
                    print(f"⚠️ Retry: Incomplete nutrition data, retrying...")
                    await asyncio.sleep(1)
                    continue
                raise ValueError("Could not extract complete nutrition data.")
            
            return text_output, parsed
        
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}, retrying...")
            await asyncio.sleep(1)
    
    raise RuntimeError("Max retries exceeded")

@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, GIF, WebP allowed.")
        
        image_bytes = await file.read()
        
        # Validate file size after reading
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_IMAGE_SIZE // 1024 // 1024}MB")
        
        base64_image, mime_type = compress_image(image_bytes)
        text_output, parsed = await _analyze_food_with_retry(base64_image, mime_type)
        
        return {
            "success": True,
            "raw_response": text_output,
            "parsed": parsed,
        }

    except HTTPException:
        raise
    except ValueError as e:
        print(f"❌ VALIDATION ERROR: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ ANALYZE ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

class SearchRequest(BaseModel):
    query: str

async def _search_food_with_retry(sanitized_query: str, max_retries: int = 2):
    """Helper function to search food with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            prompt = f"""You are an expert nutritionist with deep knowledge of all global cuisines, including Indian food.
The user searched for: "{sanitized_query}"

Provide accurate nutrition information for this specific food item.
Respond ONLY in this exact format — no extra text, no markdown, no explanation:
Dish Name: <exact name>
Estimated Portion: <standard portion>
Total Calories: <kcal number only>
Protein: <grams number only>
Carbs: <grams number only>
Fat: <grams number only>"""
            print(f"🔍 Searching AI for: {sanitized_query} (attempt {attempt + 1}/{max_retries + 1})")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            text_output = response.choices[0].message.content.strip()
            parsed = parse_nutrition_response(text_output)
            
            # Validate parsed output
            if not parsed.get("dish_name"):
                if attempt < max_retries:
                    print(f"⚠️ Retry: Could not parse dish name, retrying...")
                    await asyncio.sleep(1)
                    continue
                raise ValueError("Could not parse nutrition data. Response format invalid.")
            
            # Ensure all required fields are present
            required_fields = ["calories", "protein_g", "carbs_g", "fat_g"]
            if any(parsed.get(field) is None for field in required_fields):
                if attempt < max_retries:
                    print(f"⚠️ Retry: Incomplete nutrition data, retrying...")
                    await asyncio.sleep(1)
                    continue
                raise ValueError("Incomplete nutrition data received.")
            
            return text_output, parsed
        
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}, retrying...")
            await asyncio.sleep(1)
    
    raise RuntimeError("Max retries exceeded")

@app.post("/search-food")
async def search_food(request: SearchRequest):
    try:
        # Input validation
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(request.query) > 500:
            raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")
        
        # Sanitize input to prevent prompt injection
        sanitized_query = request.query.strip()[:500]
        
        text_output, parsed = await _search_food_with_retry(sanitized_query)
        return {"success": True, "raw_response": text_output, "parsed": parsed}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ SEARCH ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ── PANTRY ARCHITECT ENDPOINT ─────────────────────────────
class PantryRequest(BaseModel):
    ingredients: List[str]
    goal: str

@app.post("/generate-recipes")
async def generate_recipes(request: PantryRequest):
    try:
        # Validate ingredients
        if not request.ingredients or len(request.ingredients) == 0:
            raise HTTPException(status_code=400, detail="Ingredients list cannot be empty.")
        if len(request.ingredients) > 50:
            raise HTTPException(status_code=400, detail="Too many ingredients (max 50).")
        if any(len(ing.strip()) == 0 for ing in request.ingredients):
            raise HTTPException(status_code=400, detail="Ingredients cannot be empty strings.")
        
        # Validate goal
        valid_goals = ["lose", "muscle", "maintain", "athletic"]
        if request.goal not in valid_goals:
            raise HTTPException(status_code=400, detail=f"Invalid goal. Must be one of: {', '.join(valid_goals)}")
        
        ingredients_str = ", ".join(request.ingredients)

        goal_descriptions = {
            "lose":     "weight loss (low calorie, high fibre, high protein, keeps user full)",
            "muscle":   "muscle gain (very high protein, adequate complex carbs, calorie surplus)",
            "maintain": "weight maintenance (balanced macros, variety, moderation)",
            "athletic": "athletic performance (high complex carbs for energy, anti-inflammatory, fast recovery)"
        }
        goal_desc = goal_descriptions.get(request.goal, "general health")

        required_recipe_fields = {
            "rank", "emoji", "name", "description", "goalMatchPct", "goalMatchReason",
            "cal", "protein", "carbs", "fat", "fiber", "portion",
            "ingredientsUsed", "optionalBoosts", "cookingSteps", "nutritionistTweak", "satietyScore"
        }
        
        for attempt_num in range(3):  # Max 3 attempts
            try:
                prompt = f"""You are an expert nutritionist and chef specialising in Indian and global cuisines.

The user has these ingredients available: {ingredients_str}
Their fitness goal is: {goal_desc}

Generate exactly 3 recipes using ONLY these ingredients.
You may suggest maximum 1-2 optional extra ingredients per recipe.
Rank them from best to worst match for their goal.
Make the recipes realistic, practical and goal-aligned.

Respond ONLY with valid JSON and no extra text, no markdown, no explanation.
Return an object with a "recipes" key containing exactly 3 recipe objects.
Each recipe object must include these keys:
- rank (integer)
- emoji (string)
- name (string)
- description (string)
- goalMatchPct (number)
- goalMatchReason (string)
- cal (number)
- protein (number)
- carbs (number)
- fat (number)
- fiber (number)
- portion (string)
- ingredientsUsed (array of strings)
- optionalBoosts (array of strings)
- cookingSteps (array of strings)
- nutritionistTweak (string)
- satietyScore (number)"""

                print(f"🧑‍🍳 Generating recipes (attempt {attempt_num + 1}/3)")

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )

                text_output = response.choices[0].message.content.strip()
                print(f"🍽️ Raw recipe response: {text_output[:200]}...")

                # Clean JSON response - try multiple formats
                cleaned_json = text_output
                if "```json" in text_output:
                    cleaned_json = text_output.split("```json")[1].split("```")[0].strip()
                elif "```" in text_output:
                    cleaned_json = text_output.split("```")[1].split("```")[0].strip()
                
                recipes_data = json.loads(cleaned_json)
                
                # Validate structure
                if "recipes" not in recipes_data:
                    raise ValueError("JSON missing 'recipes' key.")
                
                recipes = recipes_data["recipes"]
                if not isinstance(recipes, list):
                    raise ValueError("'recipes' must be an array.")
                
                if len(recipes) != 3:
                    if attempt_num < 2:
                        print(f"⚠️ Retry: Expected 3 recipes, got {len(recipes)}, retrying...")
                        await asyncio.sleep(1)
                        continue
                    raise ValueError(f"Expected exactly 3 recipes, got {len(recipes)}.")
                
                # Validate each recipe has required fields
                all_valid = True
                for i, recipe in enumerate(recipes):
                    missing_fields = required_recipe_fields - set(recipe.keys())
                    if missing_fields:
                        if attempt_num < 2:
                            print(f"⚠️ Retry: Recipe {i+1} missing fields: {missing_fields}, retrying...")
                            await asyncio.sleep(1)
                            all_valid = False
                            break
                        raise ValueError(f"Recipe {i+1} missing fields: {missing_fields}")
                
                if all_valid:
                    print(f"✅ Successfully generated {len(recipes)} recipes!")
                    return {
                        "success": True,
                        "recipes": recipes
                    }
            
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt_num == 2:
                    print(f"❌ RECIPE ERROR: {e}")
                    raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")
                print(f"⚠️ Attempt {attempt_num + 1} failed: {str(e)}, retrying...")
                await asyncio.sleep(1)
        
        raise HTTPException(status_code=500, detail="Recipe generation max retries exceeded.")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ PANTRY ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")