import os
from dotenv import load_dotenv

load_dotenv()


EMBEDDINGS_OPENAI_API_KEY=os.getenv('EMBEDDINGS_OPENAI_API_KEY')
CHROMADB_IP=os.getenv('CHROMADB_IP')
CHROMADB_PORT=os.getenv('CHROMADB_PORT')

LLM_API_KEY=os.getenv('LLM_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')