from langdetect import detect

def detect_language(text: str) -> str:
    # Detect language
    try:
        return detect(text)
    except:
        return "vi" 