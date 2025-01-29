
# AutoTrainerX

## 📚 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [License](#-license)
- [Contributing](#-contributing)
- [Future Enhancements](#-future-enhancements)
- [Contact](#-contact)

## 🚀 Overview
AutoTrainerX is an automated AI fine-tuning pipeline that allows users to upload, process, and fine-tune models using OpenAI's GPT models. It features seamless file handling, intelligent data categorization, and a streamlined fine-tuning pipeline.

## 🎯 Features
- **Multi-File Uploads:** Supports PDF, TXT, and CSV formats.
- **Content Categorization:** Uses AI to filter and classify text.
- **Fine-Tuning Pipeline:** Converts extracted text into structured training data.
- **REST API Integration:** FastAPI-powered backend with endpoints for uploading and querying models.
- **Web Interface:** Streamlit-powered UI for user-friendly interactions.
- **Real-Time AI Querying:** Allows users to interact with fine-tuned models.
- **Robust Logging & Error Handling:** Ensures smooth operation.

## 📋 Prerequisites
- Python 3.8+
- OpenAI API Key

## 🛠️ Installation
Ensure you have Python 3.8+ installed.

```sh
# Clone the repository
git clone https://github.com/FritzHeider/AutoTrainerX.git
cd AutoTrainerX

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
### **Start FastAPI Server**
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

### **Run Streamlit UI**
```sh
streamlit run app.py
```

### **Sample API Requests**
- **Upload Files**:
```sh
curl -X POST "http://localhost:8000/upload/" -F "files=@sample.pdf"
```

- **Start Fine-Tuning Job**:
```sh
curl -X POST "http://localhost:8000/fine-tune/"
```

- **Query Model**:
```sh
curl -X POST "http://localhost:8000/query/" -d '{"prompt": "Hello, how are you?"}'
```

## 📡 API Endpoints
| Endpoint       | Method | Description                     |
| -------------- | ------ | ------------------------------- |
| `/upload/`     | `POST` | Upload and process files         |
| `/fine-tune/`  | `POST` | Start fine-tuning job            |
| `/query/`      | `POST` | Query fine-tuned model           |

## 📝 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contributing
We welcome contributions! Feel free to submit issues and pull requests.

## 🛠️ Future Enhancements
- Add support for more file formats.
- Implement real-time fine-tuning job tracking.
- Enhance AI response accuracy with embedding-based analysis.

## 📧 Contact
For support or inquiries, reach out via [GitHub Issues](https://github.com/FritzHeider/AutoTrainerX/issues).

