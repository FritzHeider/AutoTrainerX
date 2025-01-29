🚀 GPT Fine-Tuning & File Processing API - README Update

📌 Overview

This project is a high-performance AI fine-tuning & file processing system that:
✅ Supports PDF, TXT, CSV uploads.
✅ Uses OpenAI GPT-4 & GPT-3.5-Turbo for fine-tuning and queries.
✅ Stores files in AWS S3 instead of local storage.
✅ Uses PostgreSQL to track uploaded files & metadata.
✅ Secures API with JWT authentication and rate limiting.
✅ Features a modern Streamlit UI with Dark Mode, File Uploads, AI Query & Progress Bars.

📁 Features

1️⃣ API Enhancements
	•	✅ FastAPI-based Backend (Asynchronous processing with asyncio)
	•	✅ Automatic Model Selection (GPT-3.5-Turbo for short texts, GPT-4 for complex texts)
	•	✅ AWS S3 Cloud Storage (Scalable file management)
	•	✅ PostgreSQL Database Integration (Tracks file metadata)
	•	✅ JWT Authentication (Secure API access)
	•	✅ Rate Limiting (Prevents API abuse)
	•	✅ Error Handling & Logging (Structured logs in logs/app.log)

2️⃣ AI Processing
	•	✅ AI-Assisted Preprocessing (Stopword removal, text cleaning via nltk)
	•	✅ Fine-Tuning Data Formatting (Automatic conversation extraction)
	•	✅ Dynamic Model Switching (GPT-4 for large files, GPT-3.5 for smaller ones)

3️⃣ Streamlit UI
	•	✅ Dark Mode UI
	•	✅ File Upload Progress Bar
	•	✅ Real-time AI Query Interface
	•	✅ View Processed Files from AWS S3
	•	✅ Better Error Alerts in UI

🔧 Tech Stack

Component	Technology Used
🏗 Backend	FastAPI (async)
🤖 AI Model	OpenAI GPT-4 / GPT-3.5-Turbo
📂 Storage	AWS S3
🗄 Database	PostgreSQL (asyncpg)
🔑 Security	JWT Authentication
⏳ Rate Limiting	SlowAPI (5 requests/min)
🎨 Frontend	Streamlit (Dark Mode)
🏗 Deployment	Docker + AWS/GCP

🛠 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-repo/AI-Fine-Tuning-Tool.git
cd AI-Fine-Tuning-Tool

2️⃣ Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up Environment Variables

Create a .env file with:

OPENAI_API_KEY="your_openai_api_key"
AWS_BUCKET_NAME="your_s3_bucket"
AWS_ACCESS_KEY="your_aws_access_key"
AWS_SECRET_KEY="your_aws_secret_key"
DATABASE_URL="postgresql+asyncpg://user:password@localhost/dbname"
JWT_SECRET_KEY="your_jwt_secret"

5️⃣ Start Backend API

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

6️⃣ Start Frontend (Streamlit)

streamlit run ui.py

📌 API Endpoints

Method	Endpoint	Description	Auth Required?
POST	/upload/	Upload multiple files	❌
POST	/fine-tune/	Start fine-tuning process	✅
POST	/query/	Query the fine-tuned AI	✅
GET	/files/	Get list of uploaded files	❌

🚀 Usage

1️⃣ Upload Files

curl -X POST "http://localhost:8000/upload/" -F "files=@test.pdf"

2️⃣ Fine-Tune AI Model

curl -X POST "http://localhost:8000/fine-tune/" -H "Authorization: Bearer your_token"

3️⃣ Query AI Model

curl -X POST "http://localhost:8000/query/" -H "Authorization: Bearer your_token" -d '{"prompt": "Explain quantum mechanics"}'

4️⃣ View Uploaded Files

curl -X GET "http://localhost:8000/files/"

🚀 Deployment

1️⃣ Build & Run with Docker

docker build -t ai-fine-tuning .
docker run -p 8000:8000 ai-fine-tuning

2️⃣ Deploy to AWS/GCP

gcloud run deploy --image=gcr.io/YOUR_PROJECT_ID/ai-fine-tuning

📌 Next Steps
	•	🔹 Integrate Fine-Tuning Jobs Monitoring
	•	🔹 Improve AI Response Logging
	•	🔹 Add Multi-User Authentication

💡 Contributors

👨‍💻 Your Name – AI Engineer
👨‍🔧 Contributor 2 – Backend Developer

🚀 Fork & contribute to this project! 💡