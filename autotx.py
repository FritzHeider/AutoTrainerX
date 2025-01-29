```python
# Standard library imports
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

# Third-party imports
import PyPDF2
import pandas as pd
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import streamlit as st
from pydantic import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to capture detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

# Create upload directory
Path(settings.upload_dir).mkdir(exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        logger.info(f"Extracted text from PDF: {pdf_path}")
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {str(e)}")
        raise HTTPException(status_code=404, detail="PDF file not found")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error extracting text from PDF")
    return text

def load_text_file(text_path: str) -> str:
    """
    Load text from a file.

    Args:
        text_path (str): Path to the text file.

    Returns:
        str: Loaded text.
    """
    try:
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()
        logger.info(f"Loaded text from file: {text_path}")
    except FileNotFoundError as e:
        logger.error(f"Text file not found: {str(e)}")
        raise HTTPException(status_code=404, detail="Text file not found")
    except Exception as e:
        logger.error(f"Error loading text file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading text file")
    return text

def load_csv_data(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        List[Dict[str, Any]]: Loaded CSV data as a list of dictionaries.
    """
    try:
        df = pd.read_csv(csv_path)
        data = df.to_dict(orient="records")
        logger.info(f"Loaded CSV data from file: {csv_path}")
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {str(e)}")
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading CSV data")
    return data

def format_for_finetuning(text: str) -> List[Dict[str, Any]]:
    """
    Format text into fine-tuning JSONL format.

    Args:
        text (str): Input text to format.

    Returns:
        List[Dict[str, Any]]: Formatted data for fine-tuning.
    """
    formatted_data = []
    conversations = text.split("\n\n")  # Assume paragraph breaks indicate separate interactions
    for convo in conversations:
        parts = convo.split("\n", 1)
        if len(parts) == 2:
            user_prompt, response = parts
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": user_prompt.strip()},
                    {"role": "assistant", "content": response.strip()}
                ]
            })
    logger.debug(f"Formatted data for fine-tuning: {formatted_data}")
    return formatted_data

class ContentCategory(Enum):
    VALID_CONVERSATION = "valid_conversation"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    NONSENSE = "nonsense"
    IRRELEVANT = "irrelevant"
    UNCATEGORIZED = "uncategorized"

async def analyze_content(text: str) -> Tuple[ContentCategory, float, Optional[str]]:
    """
    Analyze content using LLM to determine quality and category.

    Args:
        text (str): Text to analyze.

    Returns:
        Tuple[ContentCategory, float, Optional[str]]: (category, confidence_score, explanation)
    """
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                Analyze the following text and categorize it. Determine if it contains:
                1. Valid conversation pairs (question/answer, instruction/response)
                2. Technical documentation
                3. Nonsense or gibberish
                4. Irrelevant content
                
                Respond in JSON format with:
                {
                    "category": "valid_conversation|technical_documentation|nonsense|irrelevant|uncategorized",
                    "confidence": 0.0-1.0,
                    "explanation": "Brief explanation of the categorization"
                }
                """},
                {"role": "user", "content": text[:1000]}  # First 1000 chars as sample
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Content analysis result: {result}")
        return (
            ContentCategory(result["category"]),
            float(result["confidence"]),
            result["explanation"]
        )
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}")
        return ContentCategory.UNCATEGORIZED, 0.0, str(e)

async def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file size and extension.

    Args:
        file (UploadFile): Uploaded file to validate.
    """
    if file.size > settings.max_file_size:
        logger.warning(f"File too large: {file.filename}")
        raise HTTPException(status_code=400, detail="File too large")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="Unsupported file format")

async def process_file(file: UploadFile) -> List[Dict[str, Any]]:
    """
    Process a single file and return formatted data.

    Args:
        file (UploadFile): Uploaded file to process.

    Returns:
        List[Dict[str, Any]]: Formatted data for fine-tuning.
    """
    try:
        await validate_file(file)
        file_path = Path(settings.upload_dir) / file.filename
        
        # Save file
        content = await file.read()
        await file.seek(0)
        file_path.write_bytes(content)
        logger.debug(f"Saved file: {file.filename}")

        # Process based on file type
        if file_path.suffix.lower() == '.pdf':
            extracted_text = extract_text_from_pdf(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            extracted_text = load_text_file(str(file_path))
        elif file_path.suffix.lower() == '.csv':
            extracted_text = json.dumps(load_csv_data(str(file_path)))
        
        # Analyze content quality and category
        category, confidence, explanation = await analyze_content(extracted_text)
        
        # Log analysis results
        logger.info(f"File {file.filename} analyzed: {category.value} (confidence: {confidence:.2f})")
        
        # Skip processing if content is nonsense or irrelevant
        if category in [ContentCategory.NONSENSE, ContentCategory.IRRELEVANT]:
            raise HTTPException(
                status_code=400,
                detail=f"Content rejected: {explanation}"
            )
        
        # Process content based on category
        if category == ContentCategory.TECHNICAL_DOCUMENTATION:
            # Split technical documentation into smaller chunks
            formatted_data = await split_technical_docs(extracted_text)
        else:
            formatted_data = format_for_finetuning(extracted_text)
        
        return formatted_data
    
    except HTTPException as e:
        logger.warning(f"File {file.filename} rejected: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def split_technical_docs(text: str) -> List[Dict[str, Any]]:
    """
    Split technical documentation into Q&A pairs using LLM.

    Args:
        text (str): Technical documentation text to split.

    Returns:
        List[Dict[str, Any]]: Formatted Q&A pairs for fine-tuning.
    """
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                Convert this technical documentation into a series of Q&A pairs.
                For each concept or section, create:
                1. A question or prompt that would ask for this information
                2. A clear, concise answer using the documentation content
                
                Format each pair as: Q: [question] A: [answer]
                """},
                {"role": "user", "content": text}
            ]
        )
        
        qa_text = response.choices[0].message.content
        logger.debug(f"Split technical documentation into Q&A pairs")
        return format_for_finetuning(qa_text)
    
    except Exception as e:
        logger.error(f"Error splitting technical docs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handle multiple file uploads and processing.

    Args:
        files (List[UploadFile]): List of uploaded files.

    Returns:
        dict: Response containing the status of file processing.
    """
    try:
        all_data = []
        results = {category: [] for category in ContentCategory}
        
        for file in files:
            try:
                formatted_data = await process_file(file)
                all_data.extend(formatted_data)
                
                # Store results by category
                category, _, _ = await analyze_content(formatted_data[0]["messages"][1]["content"])
                results[category].append(file.filename)
                
            except HTTPException as e:
                # Log rejected files but continue processing others
                logger.warning(f"File {file.filename} rejected: {e.detail}")
                results[ContentCategory.IRRELEVANT].append(file.filename)
                continue
        
        # Save formatted data
        finetune_file = Path("finetune_data.jsonl")
        with finetune_file.open("w") as f:
            for entry in all_data:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Successfully processed {len(files)} files")
        return {
            "message": "Files processed successfully",
            "fine_tune_data": str(finetune_file),
            "categorization": {cat.value: files for cat, files in results.items() if files}
        }
    
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune/")
async def fine_tune():
    """
    Start fine-tuning process with error handling.

    Returns:
        dict: Response containing the fine-tune job ID.
    """
    try:
        with open("finetune_data.jsonl", "rb") as file:
            response = await openai.File.acreate(
                file=file,
                purpose="fine-tune"
            )
        
        fine_tune_job = await openai.FineTuningJob.create(
            training_file=response.id,
            model="gpt-3.5-turbo"
        )
        
        logger.info(f"Started fine-tuning job: {fine_tune_job.id}")
        return {"message": "Fine-tuning started", "fine_tune_id": fine_tune_job.id}
    
    except Exception as e:
        logger.error(f"Error in fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_model(prompt: str):
    """
    Query the fine-tuned model with error handling.

    Args:
        prompt (str): Input prompt to query the model.

    Returns:
        dict: Response containing the model's answer.
    """
    try:
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Empty prompt")
        
        response = await openai.ChatCompletion.acreate(
            model=settings.fine_tuned_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        logger.error(f"Error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI
st.title("GPT Fine-Tuning Tool")

uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=["pdf", "txt", "csv"])
if uploaded_files:
    st.write("Uploaded Files:")
    try:
        with st.spinner("Processing files..."):
            response = upload_files(uploaded_files)
            
            # Display categorization results
            st.subheader("File Categorization Results")
            for category, files in response["categorization"].items():
                if files:
                    st.write(f"**{category.replace('_', ' ').title()}:**")
                    for file in files:
                        st.write(f"- {file}")
            
            st.success("Files processed successfully!")
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")

if st.button("Start Fine-Tuning"):
    try:
        with st.spinner("Fine-tuning in progress..."):
            fine_tune_response = fine_tune()
            st.write(f"Fine-Tuning ID: {fine_tune_response['fine_tune_id']}")
    except Exception as e:
        st.error(f"Error during fine-tuning: {str(e)}")

query_input = st.text_input("Ask the fine-tuned model:")
if st.button("Query Model"):
    try:
        with st.spinner("Querying model..."):
            response = query_model(query_input)
            st.write("Response:", response["response"])
    except Exception as e:
        st.error(f"Error querying model: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
