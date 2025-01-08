from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
import numpy as np

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FastAPI app
app = FastAPI()

# Keywords -  curently in the AAhaas app
keywords = [
    "hiker",
    "forest",
    "backpack",
    "walking stick",
    "sunlight",
    "tree",
    "outdoors",
    "hiking",
    "nature",
    "adventure",
    "exploration",
    "camping",
    "forest path",
    "sunset",
    "sunrise",
    "chairs",
    "trees",
    "city skyline",
    "decorative lights",
    "coffee stall",
    "outdoor seating",
    "umbrellas",
    "coffee",
    "temple",
    "mosque",
    "pagoda",
    "landmarks",
    "cultural diversity",
    "architectural styles",
    "sun",
    "architecture",
    "diverse cultures",
    "religious buildings",
    "cultural symbols",
    "flat design",
    "scenic",
    "sunny",
    "green landscape",
    "mountain range",
    "countryside",
    "house",
    "mountains",
    "greenery",
    "fields",
    "tropical",
    "beach",
    "island",
    "ocean",
    "palm trees",
    "shells",
    "starfish",
    "water",
    "sand",
    "backpacker",
    "outdoor",
    "surfing",
    "male",
    "on the water",
    "paddleboarding",
    "female",
    "on a paddleboard",
    "snorkeling",
    "underwater",
    "water sports",
    "beach activities",
    "cooking",
    "at the stove",
    "preparing ingredients",
    "at the counter",
    "baking",
    "kitchen",
    "meal preparation",
    "group activity",
    "Cartoon style",
    "Outdoor scene",
    "Wildlife",
    "Safari adventure",
    "safari",
    "wildlife",
    "elephants",
    "jeep",
    "woman",
    "spa",
    "relaxation",
    "massage",
    "back massage",
    "hot stones",
    "floral decor",
    "wellness",
    "wooden floor",
    "spa setting",
    "herbal compress",
    "tropical leaves",
    "candles",
    "spa tray",
    "wellness therapy",
    "spa treatment",
    "relaxation therapy",
    "hot stone massage",
    "floral decorations",
    "calm",
    "spa day",
    "man",
    "glamping",
    "mountain view",
    "geodesic dome",
    "cozy bed",
    "luxury camping",
    "nature retreat",
    "scenic view",
    "relaxing escape",
    "cozy stay",
    "group of people",
    "party",
    "music",
    "guitar",
    "dancing",
    "social gathering",
    "friends",
    "indoors",
    "two beds",
    "beer bottles",
    "guitars",
    "lively atmosphere",
    "casual clothing",
    "indoor party",
    "room with windows",
    "house party",
    "music jam session",
    "friends gathering",
    "guitar playing",
    "social event",
    "cabin",
    "sky",
    "clouds",
    "minimalistic",
    "mountainscape",
    "forest cabin",
    "tranquil",
    "minimalistic cabin",
    "vacation",
    "hotel",
    "rating",
    "stars",
    "customers",
    "feedback",
    "review",
    "illustration",
    "service industry",
    "hotel rating",
    "customer feedback",
    "review system",
    "service rating",
    "online reviews",
    "hospitality",
    "travel industry",
    "customer experience",
    "lounging",
    "blue swimsuit",
    "in pool",
    "swim trunks",
    "poolside",
    "daytime",
    "resort",
    "leisure",
    "summer",
    "bottle",
    "Gray",
    "purple",
    "pump",
    "White",
    "screw",
    "jar",
    "orange",
    "pink",
    "red",
    "dropper bottle",
    "brown",
    "bubbles",
    "sparkles",
    "cleaning products",
    "cosmetic containers",
    "variety of packaging",
    "bright colors",
    "hygienic and fresh atmosphere",
    "skincare",
    "cosmetics",
    "bottles",
    "cleanliness",
    "beauty products",
    "personal care"
]


# Secret key constant
SECRET_KEY = "SuperSecretKaMeyAhDS#@Aahaas"

class Description(BaseModel):
    description: str
    secret_key: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Endpoint to get keywords for a given description
@app.post("/getkey_words")
async def get_key_words(request: Description):
    # Validate secret key
    if request.secret_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid secret key")
    
    # Generate embeddings
    desc_embedding = model.encode(request.description)
    keyword_embeddings = model.encode(keywords)
    
    # Calculate similarities
    similarities = util.cos_sim(desc_embedding, keyword_embeddings)
    threshold = 0.2 #Decrease to get more keywords, Increase for more specific keywords
    matched_keywords = [
        keywords[i] 
        for i in range(len(keywords)) 
        if similarities[0][i] > threshold
    ]
    
    return {"keywords": matched_keywords}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}