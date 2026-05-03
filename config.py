import os
import configparser
from dotenv import load_dotenv

load_dotenv()

_config = configparser.ConfigParser()
_config.read("config.ini")

def _get_val(section: str, key: str, fallback: str) -> str:
    # Env var takes precedence, then config.ini, then fallback
    env_val = os.getenv(key)
    if env_val:
        return env_val
    if _config.has_section(section) and _config.has_option(section, key):
        val = _config.get(section, key)
        if val:
            return val
    return fallback

HF_TOKEN: str = _get_val("API_KEYS", "HF_TOKEN", "")
GROQ_API_KEY: str = _get_val("API_KEYS", "GROQ_API_KEY", "")
GEMINI_API_KEY: str = _get_val("API_KEYS", "GEMINI_API_KEY", "")

MEDGEMMA_MODEL_ID: str = _get_val("MODELS", "MEDGEMMA_MODEL_ID", "google/medgemma-27b-text-it:featherless-ai")
GROQ_MODEL_ID: str = _get_val("MODELS", "GROQ_MODEL_ID", "llama-3.3-70b-versatile")
GEMINI_MODEL_ID: str = _get_val("MODELS", "GEMINI_MODEL_ID", "gemini-2.5-flash")
EMBEDDING_MODEL_ID: str = _get_val("MODELS", "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

DB_PATH: str = _get_val("STORAGE", "DB_PATH", "./triageai.db")
FAISS_INDEX_PATH: str = _get_val("STORAGE", "FAISS_INDEX_PATH", "./rag/disease_index/")
ONTOLOGY_URL: str = _get_val("STORAGE", "ONTOLOGY_URL", "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/HumanDO.obo")

LOG_LEVEL: str = _get_val("SYSTEM", "LOG_LEVEL", "INFO")
CONFIDENCE_THRESHOLD_LOW: float = float(_get_val("SYSTEM", "CONFIDENCE_THRESHOLD_LOW", "0.55"))
CONFIDENCE_THRESHOLD_CRITICAL: float = float(_get_val("SYSTEM", "CONFIDENCE_THRESHOLD_CRITICAL", "0.40"))
API_TIMEOUT_SECONDS: int = int(_get_val("SYSTEM", "API_TIMEOUT_SECONDS", "45"))
MAX_LLM_RETRIES: int = int(_get_val("SYSTEM", "MAX_LLM_RETRIES", "1"))
RAG_TOP_K: int = int(_get_val("SYSTEM", "RAG_TOP_K", "5"))
BENCHMARK_TIMEOUT_SECONDS: int = int(_get_val("SYSTEM", "BENCHMARK_TIMEOUT_SECONDS", "300"))
STREAMLIT_API_BASE_URL: str = _get_val("SYSTEM", "STREAMLIT_API_BASE_URL", "http://localhost:8000")
APP_VERSION: str = _get_val("SYSTEM", "APP_VERSION", "1.0.0")
APP_NAME: str = _get_val("SYSTEM", "APP_NAME", "TriageAI")
