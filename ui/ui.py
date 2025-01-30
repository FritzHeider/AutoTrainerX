# ui/ui.py
import streamlit as st
from core.trainer import train_model

def main():
    st.title("AutoTrainerX UI")
    model_type = st.selectbox("Select Model Type", ["Neural Network", "Decision Tree", "SVM"])
    dataset_path = st.text_input("Dataset Path")
    hyperparameters = st.text_area("Hyperparameters (JSON format)")
    
    if st.button("Train Model"):
        result = train_model(model_type, dataset_path, eval(hyperparameters))
        st.success(result)

if __name__ == "__main__":
    main()
