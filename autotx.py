# Standard library imports
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

# Third-party imports
import aiofiles
import httpx
import PyPDF2
import pandas as pd
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re

# Configure structured logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# NLP Preprocessing Setup
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Cleans input text using NLP techniques."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

# AI Model Selection
def select_model(text: str) -> str:
    """Selects GPT-3.5-Turbo or GPT-4 based on input size and complexity."""
    token_count = len(text.split())
    return "gpt-4" if token_count > 1500 else "gpt-3.5-turbo"

async def fetch_with_retries(api_call, retries=3, delay=2):
    """Retries API calls with exponential backoff."""
    for attempt in range(retries):
        try:
            return await api_call()
        except openai.error.OpenAIError as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))
            else:
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def analyze_content(text: str) -> Tuple[str, float, Optional[str]]:
    """Uses OpenAI to analyze and categorize content."""
    model = select_model(text)
    text = clean_text(text)

    async with httpx.AsyncClient() as client:
        response = await fetch_with_retries(lambda: client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Classify the text into conversation, technical docs, nonsense, or irrelevant."},
                    {"role": "user", "content": text[:1500]}
                ]
            }
        ))

    result = response.json()
    return result["category"], float(result["confidence"]), result["explanation"]

@app.post("/query/")
async def query_model(prompt: str):
    """Query the fine-tuned model with dynamic model selection."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    model = select_model(prompt)

    async with httpx.AsyncClient() as client:
        response = await fetch_with_retries(lambda: client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]}
        ))

    return {"response": response.json()["choices"][0]["message"]["content"]}