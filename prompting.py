import langchain
import predictionguard as pg
import huggingface_hub
import datasets

from huggingface_hub import login
from datasets import load_dataset

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from the environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

login(token="YOUR_HF_TOKEN")  


dataset = load_dataset("dataset_name")  