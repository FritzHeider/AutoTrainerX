# Standard library imports
import os
import json
import time
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
from pydantic import BaseSettings

# Configure structured logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Configuration class
class Settings(BaseSettings):
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".pdf", ".txt", ".csv"]
    fine_tuned_model: str = "ft:gpt-3.5-turbo-custom"

    class Config:
        env_file = ".env"

settings = Settings()

# Initialize FastAPI app
app = FastAPI()

# Ensure upload directory exists
Path(settings.upload_dir).mkdir(exist_ok=True)

async def extract_text_from_pdf(pdf_path: str) -> str:
    """ Extracts text from a PDF asynchronously using streaming."""
    text = ""
    try:
        async with aiofiles.open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(await file.read())
            for page in reader.pages:
                text += page.extract_text() + "\n"
        logger.info(f"Extracted text from PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error extracting text from PDF")
    return text

async def load_text_file(text_path: str) -> str:
    """ Loads text from a file asynchronously. """
    try:
        async with aiofiles.open(text_path, "r", encoding="utf-8") as file:
            text = await file.read()
        logger.info(f"Loaded text from file: {text_path}")
    except Exception as e:
        logger.error(f"Error loading text file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading text file")
    return text

async def load_csv_data(csv_path: str) -> List[Dict[str, Any]]:
    """ Loads CSV data asynchronously without blocking. """
    try:
        async with aiofiles.open(csv_path, "r", encoding="utf-8") as file:
            content = await file.read()
        df = pd.read_csv(pd.io.StringIO(content))
        data = df.to_dict(orient="records")
        logger.info(f"Loaded CSV data from file: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading CSV data")
    return data

async def fetch_with_retries(api_call, retries=3, delay=2):
    """ Wrapper for API calls with exponential backoff. """
    for attempt in range(retries):
        try:
            return await api_call()
        except openai.error.OpenAIError as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))
            else:
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def analyze_content(text: str) -> Tuple[str, float, Optional[str]]:
    """ Uses OpenAI to analyze content type asynchronously. """
    async with httpx.AsyncClient() as client:
        response = await fetch_with_retries(lambda: client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Classify the following text into valid_conversation, technical_documentation, nonsense, or irrelevant."},
                    {"role": "user", "content": text[:1000]}  # Limit to first 1000 chars
                ]
            }
        ))
    result = response.json()
    return result["category"], float(result["confidence"]), result["explanation"]

async def validate_file(file: UploadFile):
    """ Validates file size and format asynchronously. """
    if file.size > settings.max_file_size:
        raise HTTPException(status_code=400, detail="File too large")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format")

async def process_file(file: UploadFile) -> List[Dict[str, Any]]:
    """ Processes a file asynchronously, returning fine-tuning data. """
    await validate_file(file)
    file_path = Path(settings.upload_dir) / file.filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    if file_path.suffix == ".pdf":
        extracted_text = await extract_text_from_pdf(str(file_path))
    elif file_path.suffix == ".txt":
        extracted_text = await load_text_file(str(file_path))
    elif file_path.suffix == ".csv":
        extracted_text = json.dumps(await load_csv_data(str(file_path)))

    category, confidence, explanation = await analyze_content(extracted_text)
    if category in ["nonsense", "irrelevant"]:
        raise HTTPException(status_code=400, detail=f"Content rejected: {explanation}")

    return [{"messages": [{"role": "user", "content": extracted_text}]}]

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """ Handles multiple file uploads asynchronously. """
    results = []
    tasks = [process_file(file) for file in files]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for file, result in zip(files, responses):
        if isinstance(result, Exception):
            logger.error(f"File {file.filename} failed: {result}")
            continue
        results.extend(result)

    async with aiofiles.open("finetune_data.jsonl", "w") as f:
        for entry in results:
            await f.write(json.dumps(entry) + "\n")

    return {"message": "Files processed successfully", "data_count": len(results)}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)