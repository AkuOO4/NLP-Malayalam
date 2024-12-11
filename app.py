from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Import the functions from your main script
from model import preprocess_input, predict, calculate_perplexity_ml

app = FastAPI()

class PredictionRequest(BaseModel):
    sentence: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Malayalam Next Word Prediction API!"}

@app.post("/predict/")
def predict_next_word(request: PredictionRequest):
    sentence = request.sentence
    
    # Preprocess the input sentence
    tokens = preprocess_input(sentence)
    if not tokens:
        return {"error": "Please provide a valid Malayalam sentence."}

    # Generate predictions
    predictions = predict(" ".join(tokens))

    # Calculate perplexity
    perplexity = calculate_perplexity_ml([sentence])

    # Handle infinite perplexity gracefully
    perplexity_score = perplexity if perplexity != float('inf') else "undefined"

    return {
        "predictions": predictions,
        "perplexity_score": perplexity_score
    }
