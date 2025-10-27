import os
import json
import joblib
from typing import Tuple, Any, Dict

MODELS_DIR = os.getenv("MODELS_DIR", "models")


def _paths(symbol: str, tag: str) -> Tuple[str, str]:
    """
    Tạo đường dẫn tuyệt đối cho model và meta theo cặp (symbol, tag).
    Ví dụ: VCB_gap.pkl / VCB_gap.json
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    return (
        os.path.join(MODELS_DIR, f"{symbol}_{tag}.pkl"),
        os.path.join(MODELS_DIR, f"{symbol}_{tag}.json"),
    )


def save_model_meta(symbol: str, tag: str, model: Any, meta: Dict) -> Tuple[str, str]:
    """
    Lưu model (joblib) và metadata (json) sau quá trình train.
    meta sẽ chứa thông tin:
      - order/trend của SARIMAX
      - scaler, feature_cols
      - metrics (rmse/mae/aic)
      - dự báo bước tới (ret_hat_next, next_price_est)
    """
    mpath, jpath = _paths(symbol, tag)
    joblib.dump(model, mpath)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return mpath, jpath


def load_model_meta(symbol: str, tag: str) -> Tuple[Any, Dict]:
    """
    Đọc model + metadata từ đĩa.
    Nếu chưa có thì trả về (None, None) để caller tự train.
    """
    mpath, jpath = _paths(symbol, tag)
    if not (os.path.exists(mpath) and os.path.exists(jpath)):
        return None, None

    model = joblib.load(mpath)
    with open(jpath, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta
