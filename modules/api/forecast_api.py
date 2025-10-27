from modules.ML.pipeline import smart_predict, predict_next_session
from modules.ML.pipeline import direction_from_return  # nếu cần reuse
from modules.api.time_api import get_now
from modules.api.stock_api import DATE_FMT  # dùng lại format ngày
from typing import Dict, Any

def _safe_get(d: Dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default

def format_forecast_brief(symbol: str) -> str:
    """
    Mô tả ngắn gọn xu hướng cho symbol (ví dụ dùng khi intent='market').
    - Nếu đang trong phiên: nói dòng tiền / xu hướng ngắn hạn
    - Nếu ngoài giờ: nói dự báo gap mở cửa phiên tới
    Trả string tiếng Việt kèm cảnh báo tham khảo.
    """
    try:
        pack = smart_predict(symbol)
    except Exception:
        # fallback nhẹ nếu model lỗi
        return ""

    sym = symbol.upper()
    mode = pack.get("mode")

    if mode == "in_session":
        dir_ = _safe_get(pack, "next_step_dir", "khó xác định")
        conf = _safe_get(pack, "step_confidence", "low")
        return (
            f"\n🔎 Dòng tiền {sym}: thị trường đang giao dịch."
            f" Xu hướng rất ngắn hạn: **{dir_}**"
            f" (độ tin cậy {conf})."
        )

    # out_of_session / next_session
    # Ưu tiên dự báo mở cửa phiên tới
    open_direction = pack.get("open_direction")
    open_gap_pct = pack.get("open_gap_pct")
    if open_direction is not None and open_gap_pct is not None:
        return (
            f"\n🔎 Dự báo phiên tới của {sym}: khả năng **{open_direction}**"
            f" khoảng {open_gap_pct:+.2f}%. (mang tính tham khảo)"
        )

    # Hoặc dự báo PM (phiên chiều)
    pm_direction = pack.get("pm_direction")
    pm_gap_pct = pack.get("pm_gap_pct")
    if pm_direction is not None and pm_gap_pct is not None:
        return (
            f"\n🔎 Dự báo phiên chiều của {sym}: có thể **{pm_direction}**"
            f" khoảng {pm_gap_pct:+.2f}%. (tham khảo)"
        )

    return ""


def format_forecast_text(symbol: str, pack: Dict[str, Any]) -> str:
    """
    Trả lời chi tiết khi user hỏi dự báo (intent='forecast').
    Đây là bản verbose (giải thích có AM/PM, dải giá, % thay đổi...).
    """
    sym = symbol.upper()
    mode = pack.get("mode", "")

    # 1. đang trong phiên
    if mode == "in_session":
        session = pack.get("session", "AM/PM")
        dir_ = pack.get("next_step_dir") or "khó xác định"
        conf = pack.get("step_confidence", "low")
        last_px = pack.get("last_px", None)
        path_pred = pack.get("path_pred", None)

        msg = [
            f"📈 Dự báo ngắn hạn nội phiên cho {sym} ({session}):",
            f"- Xu hướng kế tiếp: **{dir_}** (độ tin cậy {conf})."
        ]
        if last_px:
            msg.append(f"- Giá hiện tại khoảng ~{last_px:,.2f} VNĐ.")
        if path_pred is not None and hasattr(path_pred, 'tolist'):
            seq = path_pred.tolist()
            if seq:
                msg.append(
                    "- Quỹ đạo dự kiến (3 bước kế tiếp): "
                    + ", ".join(f"{p:,.2f} VNĐ" for p in seq)
                    + " (tham khảo)."
                )
        msg.append("⚠️ Đây không phải khuyến nghị mua/bán.")
        return "\n".join(msg)

    # 2. nếu là next_session (tức là dự báo AM hoặc PM kế tiếp)
    next_sess = pack.get("next_session")
    if mode == "next_session" and next_sess == "AM":
        band = pack.get("open_band", {})
        px_mean = band.get("px_mean")
        px_lo = band.get("px_lo")
        px_hi = band.get("px_hi")
        direction = pack.get("open_direction", "dao động nhẹ")
        gap_pct = pack.get("open_gap_pct", 0.0)
        conf = pack.get("open_confidence", "uncertain")
        target_day = pack.get("target_day")

        msg = [
            f"📅 Phiên sáng (AM) sắp tới của {sym} ({target_day}):",
            f"- Giá mở cửa dự kiến khoảng {px_mean:,.2f} VNĐ "
            f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
            f"- Dự kiến {direction} khoảng {gap_pct:+.2f}%.",
            f"- Mức độ tự tin mô hình: {conf}.",
            "⚠️ Đây chỉ là ước lượng dựa trên tin tức & hành vi giá gần nhất,"
            " không phải khuyến nghị đầu tư."
        ]
        return "\n".join(msg)

    if mode == "next_session" and next_sess == "PM":
        band = pack.get("pm_band", {})
        px_mean = band.get("px_mean")
        px_lo = band.get("px_lo")
        px_hi = band.get("px_hi")
        direction = pack.get("pm_direction", "dao động nhẹ")
        gap_pct = pack.get("pm_gap_pct", 0.0)
        conf = pack.get("pm_confidence", "uncertain")
        target_day = pack.get("target_day")

        msg = [
            f"📅 Phiên chiều (PM) kế tiếp của {sym} ({target_day}):",
            f"- Giá tham chiếu đầu phiên chiều dự kiến quanh {px_mean:,.2f} VNĐ "
            f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
            f"- Khuynh hướng {direction} khoảng {gap_pct:+.2f}%.",
            f"- Độ tin cậy: {conf}.",
            "⚠️ Đây là thông tin tham khảo, không phải lời khuyên giao dịch."
        ]
        return "\n".join(msg)

    # 3. nếu là out_of_session (dự báo phiên tới tổng quát)
    open_band = (
        pack.get("bands", {})
        .get("OPEN_am", {})
    )
    px_mean = open_band.get("px_mean")
    px_lo = open_band.get("px_lo")
    px_hi = open_band.get("px_hi")
    open_dir = pack.get("open_direction", "dao động nhẹ")
    open_gap_pct = pack.get("open_gap_pct", 0.0)
    conf = pack.get("open_confidence", "uncertain")
    target_day = pack.get("target_day")

    msg = [
        f"📅 Phiên tiếp theo của {sym} ({target_day}):",
        f"- Mở cửa dự kiến quanh {px_mean:,.2f} VNĐ "
        f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
        f"- Xu hướng khả năng {open_dir} khoảng {open_gap_pct:+.2f}%",
        "  so với giá đóng cửa gần nhất.",
        f"- Mức tự tin mô hình: {conf}.",
        "⚠️ Đây chỉ là mô phỏng thống kê, KHÔNG phải khuyến nghị mua/bán."
    ]
    return "\n".join(msg)


def get_forecast_brief_for_symbol(symbol: str) -> str:
    """
    Helper cho router.intent='market'.
    Trả string ngắn gọn nối phía sau market summary / stock info.
    """
    try:
        return format_forecast_brief(symbol)
    except Exception:
        return ""


def get_full_forecast_answer(symbol: str) -> str:
    """
    Helper cho router.intent='forecast'.
    Trả câu trả lời hoàn chỉnh (dài), safe để gửi cho user.
    """
    try:
        pack = smart_predict(symbol)
    except Exception:
        # fallback nếu smart_predict lỗi, thử dự báo phiên tới cơ bản
        pack = predict_next_session(symbol)

    return format_forecast_text(symbol, pack)
