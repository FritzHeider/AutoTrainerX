
# setup.py
from setuptools import setup, find_packages

setup(
    name="AutoTrainerX",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "streamlit",
        "pydantic",
        "pytest",
        "httpx",
        "pyyaml"
    ],
)
