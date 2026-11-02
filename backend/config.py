import os

from dotenv import load_dotenv

load_dotenv()

# Elastic Cloud
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# TokenRouter API
TOKENROUTER_API_KEY = os.getenv("TOKENROUTER_API_KEY")
TOKENROUTER_BASE_URL = os.getenv(
    "TOKENROUTER_BASE_URL",
    "https://api.tokenrouter.com/v1",
)
TOKENROUTER_MODEL = os.getenv(
    "TOKENROUTER_MODEL",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
)