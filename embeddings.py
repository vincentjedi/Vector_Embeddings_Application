# embeddings.py
from openai import AzureOpenAI
import numpy as np
from typing import List
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv


load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",  # Use latest stable version
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic"""
        response = self.client.embeddings.create(
            input=text,
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        )
        return response.data[0].embedding

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Process multiple texts efficiently"""
        response = self.client.embeddings.create(
            input=texts,
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        )
        return [data.embedding for data in response.data]