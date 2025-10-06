import logging
import os

# LOGGING SETUP

LOG_FILE = os.getenv("LOG_FILE", "rag_pipeline.log")
LOG_LEVEL = logging.DEBUG  # bật toàn bộ cấp độ log (sẽ lọc ở hàm log_debug)
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Tạo thư mục logs nếu chưa có
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if "/" in LOG_FILE else None

# Cấu hình logging (chỉ khởi tạo một lần)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger("RAGPipeline")

# LOGGING HELPERS

DEBUG_MODE = os.getenv("DEBUG_MODE", "1") == "1"  # True nếu ENV=1


def log_debug(msg: str, state=None):
    """
    Log message cấp DEBUG (chỉ hiện khi DEBUG_MODE hoặc state.debug = True)
    """
    if DEBUG_MODE or (state and getattr(state, "debug", False)):
        logger.debug(msg)


def log_info(msg: str):
    """
    Log message cấp INFO — luôn hiển thị.
    """
    logger.info(msg)


def log_error(msg: str, exc: Exception = None):
    """
    Log message cấp ERROR (ghi kèm exception nếu có).
    """
    if exc:
        logger.error(f"{msg} | {type(exc).__name__}: {exc}")
    else:
        logger.error(msg)
