import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# This tells the bouncer to let everyone in
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the user's comment. "
                        "Reply ONLY with a valid JSON object, no extra text. "
                        "Use this exact format: {\"sentiment\": \"positive\", \"rating\": 5} "
                        "sentiment must be one of: positive, negative, neutral. "
                        "rating must be an integer 1-5 (1=very negative, 3=neutral, 5=very positive)."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        if data["sentiment"] not in ["positive", "negative", "neutral"]:
            raise ValueError("Invalid sentiment value")
        if not (1 <= int(data["rating"]) <= 5):
            raise ValueError("Rating out of range")
        return SentimentResponse(sentiment=data["sentiment"], rating=int(data["rating"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)