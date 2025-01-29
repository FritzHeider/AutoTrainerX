import streamlit as st
import requests
import time

st.set_page_config(page_title="GPT Fine-Tuning Tool", page_icon="🤖", layout="wide")

st.title("📄 AI Fine-Tuning & File Processing")

st.sidebar.header("⚙️ Settings")
st.sidebar.text("Dark Mode Enabled 🌙")

# Upload Files
uploaded_files = st.file_uploader("📤 Upload Files", accept_multiple_files=True, type=["pdf", "txt", "csv"])

if uploaded_files:
    st.write("📂 **Processing Files...**")
    progress_bar = st.progress(0)
    
    uploaded_data = []
    
    for idx, file in enumerate(uploaded_files):
        files = {"files": (file.name, file.getvalue())}
        response = requests.post("http://localhost:8000/upload/", files=files)
        
        if response.status_code == 200:
            uploaded_data.append(response.json())
            progress_bar.progress((idx + 1) / len(uploaded_files))
        else:
            st.error(f"❌ Failed to upload {file.name}: {response.json()['detail']}")

    st.success("✅ All files processed successfully!")

# Display Uploaded Files
st.subheader("📊 Uploaded File History")

uploaded_files = requests.get("http://localhost:8000/files").json()
if uploaded_files:
    for file in uploaded_files:
        st.write(f"📄 {file['filename']} - 🔗 [View File]({file['s3_url']})")
else:
    st.write("No uploaded files found.")

# Query AI Model
st.subheader("💬 Query Fine-Tuned Model")
query_input = st.text_input("🔍 Ask the AI Model:")

if st.button("🧠 Query AI"):
    if query_input.strip():
        with st.spinner("Thinking... 🤔"):
            response = requests.post("http://localhost:8000/query/", json={"prompt": query_input})
            if response.status_code == 200:
                st.success("✅ AI Response:")
                st.write(response.json()["response"])
            else:
                st.error("❌ Error querying model.")
    else:
        st.warning("⚠️ Please enter a query.")