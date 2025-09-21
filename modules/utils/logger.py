import logging
import os

LOG_FILE = os.getenv("LOG_FILE", "rag_pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RAGPipeline")

# Helper bật/tắt debug
DEBUG_MODE = os.getenv("DEBUG_MODE", "1") == "1"

def log_debug(msg: str):
    if DEBUG_MODE:
        logger.info(msg)
