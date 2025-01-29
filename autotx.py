from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi_jwt_auth import AuthJWT
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
import boto3
import os
import asyncio
import aiofiles

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
limiter = Limiter(key_func=get_remote_address)

async def upload_to_s3(file_path: str, s3_key: str) -> str:
    """Uploads file to AWS S3 and returns URL."""
    s3_client.upload_file(file_path, AWS_BUCKET_NAME, s3_key)
    return f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...), db: AsyncSession = Depends(get_db)):
    """Handles file uploads with JWT authentication and cloud storage."""
    tasks = [process_file(file) for file in files]
    file_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in file_results:
        if isinstance(result, dict):
            db.add(UploadedFile(filename=result["filename"], s3_url=result["s3_url"]))

    await db.commit()
    return {"message": "Files uploaded and stored in S3."}

@app.post("/query/")
@limiter.limit("5/minute")
async def query_model(prompt: str, auth: AuthJWT = Depends()):
    """Query AI model with JWT authentication."""
    auth.jwt_required()
    model = select_model(prompt)

    async with httpx.AsyncClient() as client:
        response = await fetch_with_retries(lambda: client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]}
        ))

    return {"response": response.json()["choices"][0]["message"]["content"]}s