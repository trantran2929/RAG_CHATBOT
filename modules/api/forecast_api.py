from typing import Dict, Any, Optional

from modules.ML.pipeline import smart_predict, predict_next_session, direction_from_return
from modules.api.time_api import get_now
from modules.api.stock_api import DATE_FMT


def _safe_get(d: Dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


# ============================================================
# HELPER: MÔ TẢ PHIÊN KẾ TIẾP (NGẮN / DÀI)
# ============================================================

def _format_next_session_brief(sym: str, pack: Dict[str, Any]) -> str:
    """
    Helper: mô tả rất ngắn cho phiên giao dịch kế tiếp (AM/PM).
    Ưu tiên dùng pack từ predict_next_session().
    """
    if not isinstance(pack, dict):
        return ""

    mode = pack.get("mode")
    next_sess = pack.get("next_session")
    target_day = pack.get("target_day")

    # Case: gói next_session (predict_next_session)
    if mode == "next_session" and next_sess in ("AM", "PM"):
        if next_sess == "AM":
            open_dir = pack.get("open_direction")
            open_gap_pct = pack.get("open_gap_pct")
            if open_dir is not None and open_gap_pct is not None:
                return (
                    f"🔮 Phiên sáng (AM) sắp tới của {sym} ({target_day}): "
                    f"khả năng **{open_dir}** khoảng {open_gap_pct:+.2f}% (tham khảo)."
                )
        else:  # PM
            pm_dir = pack.get("pm_direction")
            pm_gap_pct = pack.get("pm_gap_pct")
            if pm_dir is not None and pm_gap_pct is not None:
                return (
                    f"🔮 Phiên chiều (PM) kế tiếp của {sym} ({target_day}): "
                    f"có thể **{pm_dir}** khoảng {pm_gap_pct:+.2f}% (tham khảo)."
                )

    # Case: gói out_of_session (predict_tomorrow_full_exog)
    open_dir = pack.get("open_direction")
    open_gap_pct = pack.get("open_gap_pct")
    if open_dir is not None and open_gap_pct is not None:
        return(
            f"🔮 Phiên tới của {sym} ({target_day}): "
            f"khả năng **{open_dir}** khoảng {open_gap_pct:+.2f}% (tham khảo)."
        )

    return ""


def _format_next_session_verbose(sym: str, pack: Dict[str, Any]) -> str:
    """
    Helper: mô tả CHI TIẾT cho phiên giao dịch kế tiếp (AM/PM).
    Dùng chung cho cả trong phiên (phần 2) và ngoài phiên.
    """
    if not isinstance(pack, dict):
        return ""

    mode = pack.get("mode", "")
    next_sess = pack.get("next_session")
    target_day = pack.get("target_day")

    # 1) Gói next_session chuẩn
    if mode == "next_session" and next_sess in ("AM", "PM"):
        if next_sess == "AM":
            band = pack.get("open_band", {}) or {}
            px_mean = band.get("px_mean")
            px_lo = band.get("px_lo")
            px_hi = band.get("px_hi")
            direction = pack.get("open_direction", "dao động nhẹ")
            gap_pct = pack.get("open_gap_pct", 0.0)
            conf = pack.get("open_confidence", "uncertain")

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

        if next_sess == "PM":
            band = pack.get("pm_band", {}) or {}
            px_mean = band.get("px_mean")
            px_lo = band.get("px_lo")
            px_hi = band.get("px_hi")
            direction = pack.get("pm_direction", "dao động nhẹ")
            gap_pct = pack.get("pm_gap_pct", 0.0)
            conf = pack.get("pm_confidence", "uncertain")

            msg = [
                f"📅 Phiên chiều (PM) kế tiếp của {sym} ({target_day}):",
                f"- Giá tham chiếu đầu phiên chiều dự kiến quanh {px_mean:,.2f} VNĐ "
                f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
                f"- Khuynh hướng {direction} khoảng {gap_pct:+.2f}%.",
                f"- Độ tin cậy: {conf}.",
                "⚠️ Đây là thông tin tham khảo, không phải lời khuyên giao dịch."
            ]
            return "\n".join(msg)

    # 2) Gói out_of_session (predict_tomorrow_full_exog)
    if mode == "out_of_session":
        open_band = (pack.get("bands", {}) or {}).get("OPEN_am", {}) or {}
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

    return ""


# ============================================================
# 1) FORECAST BRIEF – DÙNG CHO INTENT 'market'
# ============================================================

def format_forecast_brief(symbol: str) -> str:
    """
    Mô tả ngắn gọn xu hướng cho symbol (ví dụ dùng khi intent='market').

    Thiết kế “auto”:
    - Nếu đang trong phiên: nói bước rất ngắn hạn + kèm thêm 1 câu ngắn về phiên tiếp theo.
    - Nếu ngoài giờ: chỉ nói phiên giao dịch kế tiếp.
    """
    sym = symbol.upper()

    try:
        pack = smart_predict(symbol)
    except Exception:
        # fallback nhẹ nếu model lỗi → thử luôn next_session
        try:
            next_pack = predict_next_session(symbol)
        except Exception:
            return ""
        brief_next = _format_next_session_brief(sym, next_pack)
        return ("\n" + brief_next) if brief_next else ""

    mode = pack.get("mode")

    # ĐANG TRONG PHIÊN → intraday + short next_session
    if mode == "in_session":
        dir_ = _safe_get(pack, "next_step_dir", "khó xác định")
        conf = _safe_get(pack, "step_confidence", "low")
        session = pack.get("session", "AM/PM")

        parts = [
            f"\n🔎 Dòng tiền {sym}: đang giao dịch phiên {session}.",
            f" Xu hướng rất ngắn hạn: **{dir_}** (độ tin cậy {conf})."
        ]

        # kèm thêm dự báo cho phiên giao dịch kế tiếp
        brief_next = ""
        try:
            next_pack = predict_next_session(symbol)
            brief_next = _format_next_session_brief(sym, next_pack)
        except Exception:
            brief_next = ""

        if brief_next:
            parts.append("\n" + brief_next)

        return "".join(parts)

    # NGOÀI GIỜ → chỉ phiên tiếp theo
    brief_next = _format_next_session_brief(sym, pack)
    if brief_next:
        return "\n" + brief_next

    return ""


def get_forecast_brief_for_symbol(symbol: str) -> str:
    """
    Helper cho router.intent='market'.
    Trả string ngắn gọn nối phía sau market summary / stock info.
    """
    try:
        return format_forecast_brief(symbol)
    except Exception:
        return ""


# ============================================================
# 2) FORECAST DETAIL – AUTO MODE (GỘP)
# ============================================================

def format_forecast_text(
    symbol: str,
    pack: Dict[str, Any],
    next_session_pack: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Dùng cho chế độ AUTO (get_full_forecast_answer):

    - Nếu ĐANG TRONG PHIÊN:
        + Phần 1: Dự báo ngắn hạn nội phiên (bước tiếp theo).
        + Phần 2: Dự báo cho PHIÊN GIAO DỊCH KẾ TIẾP (AM/PM tiếp theo).
    - Nếu NGOÀI PHIÊN:
        + Chỉ hiển thị dự báo cho phiên giao dịch kế tiếp.
    """
    sym = symbol.upper()
    mode = pack.get("mode", "")

    # 1. ĐANG TRONG PHIÊN → bước kế tiếp + phiên kế tiếp
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
        if path_pred is not None and hasattr(path_pred, "tolist"):
            seq = path_pred.tolist()
            if seq:
                msg.append(
                    "- Quỹ đạo dự kiến (3 bước kế tiếp): "
                    + ", ".join(f"{p:,.2f} VNĐ" for p in seq)
                    + " (tham khảo)."
                )
        msg.append("⚠️ Đây không phải khuyến nghị mua/bán.")
        intraday_text = "\n".join(msg)

        # Thêm phần dự báo cho phiên giao dịch kế tiếp
        ns_pack = next_session_pack
        if not isinstance(ns_pack, dict):
            try:
                ns_pack = predict_next_session(symbol)
            except Exception:
                ns_pack = None

        ns_text = _format_next_session_verbose(sym, ns_pack) if ns_pack else ""
        if ns_text:
            return intraday_text + "\n\n" + ns_text
        return intraday_text

    # 2. NGOÀI PHIÊN → chỉ hiển thị phiên giao dịch kế tiếp
    next_text = _format_next_session_verbose(sym, pack)
    if next_text:
        return next_text

    # 3. Fallback cuối (trường hợp pack không đúng schema)
    open_band = (
        pack.get("bands", {}) or {}
    ).get("OPEN_am", {}) or {}
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


def get_full_forecast_answer(symbol: str) -> str:
    """
    Helper chế độ AUTO (giữ tương thích cũ):

    - Nếu đang trong phiên: trả cả *bước tiếp theo* + *phiên giao dịch kế tiếp*.
    - Nếu ngoài phiên: chỉ trả *phiên giao dịch kế tiếp*.
    """
    main_pack: Optional[Dict[str, Any]] = None
    next_pack: Optional[Dict[str, Any]] = None

    try:
        main_pack = smart_predict(symbol)
    except Exception:
        main_pack = None

    # Nếu đang trong phiên → cố gắng lấy thêm gói next_session
    if isinstance(main_pack, dict) and main_pack.get("mode") == "in_session":
        try:
            next_pack = predict_next_session(symbol)
        except Exception:
            next_pack = None

    # Nếu ngoài phiên hoặc smart_predict lỗi → dùng luôn predict_next_session làm main_pack
    if main_pack is None or main_pack.get("mode") != "in_session":
        try:
            main_pack = predict_next_session(symbol)
        except Exception:
            return (
                f"Hiện chưa thể dự báo cho mã {symbol.upper()} do lỗi mô hình."
                " Bạn vui lòng thử lại sau."
            )

    return format_forecast_text(symbol, main_pack, next_session_pack=next_pack)


# ============================================================
# 3) API CHO ROUTER: TÁCH RÕ 2 TRƯỜNG HỢP USER HỎI
# ============================================================

def get_intraday_step_forecast_answer(symbol: str) -> str:
    """
    Dự đoán *bước tiếp theo trong phiên hiện tại* cho mã cổ phiếu.

    - Nếu đang trong phiên: trả về block nội phiên.
    - Nếu đang ngoài phiên: báo lại cho user (KHÔNG tự động nhảy sang phiên tới).
    """
    sym = symbol.upper()
    try:
        pack = smart_predict(symbol)
    except Exception:
        return (
            f"Hiện không lấy được dự báo nội phiên cho mã {sym} do lỗi mô hình."
            " Bạn vui lòng thử lại sau."
        )

    if pack.get("mode") != "in_session":
        return (
            f"{sym} hiện đang ngoài giờ giao dịch, không dự đoán được *bước tiếp theo trong phiên*.\n"
            f"Bạn có thể hỏi: \"dự đoán phiên tới của {sym}?\" để xem dự báo cho phiên giao dịch kế tiếp."
        )

    # --- copy logic nội phiên ---
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
    if path_pred is not None and hasattr(path_pred, "tolist"):
        seq = path_pred.tolist()
        if seq:
            msg.append(
                "- Quỹ đạo dự kiến (3 bước kế tiếp): "
                + ", ".join(f"{p:,.2f} VNĐ" for p in seq)
                + " (tham khảo)."
            )
    msg.append("⚠️ Đây không phải khuyến nghị mua/bán.")
    return "\n".join(msg)


def get_next_session_forecast_answer(symbol: str) -> str:
    """
    Dự báo cho *phiên giao dịch kế tiếp* (AM/PM) của mã cổ phiếu.

    - Luôn chỉ nói về phiên sắp tới (KHÔNG nói bước tiếp theo nội phiên).
    - Dùng được cả khi đang trong phiên hay ngoài phiên.
    """
    sym = symbol.upper()

    pack: Optional[Dict[str, Any]] = None

    # Ưu tiên dùng predictor chuyên cho phiên kế tiếp
    try:
        pack = predict_next_session(symbol)
    except Exception:
        # fallback: dùng smart_predict nếu nó trả mode='next_session' hoặc 'out_of_session'
        try:
            alt = smart_predict(symbol)
        except Exception:
            alt = None

        if isinstance(alt, dict) and alt.get("mode") in ("next_session", "out_of_session"):
            pack = alt

    if not isinstance(pack, dict):
        return (
            f"Hiện chưa thể dự báo phiên giao dịch kế tiếp cho mã {sym} do lỗi mô hình."
            " Bạn vui lòng thử lại sau."
        )

    text = _format_next_session_verbose(sym, pack)
    if not text:
        return (
            f"Hiện chưa thể lấy được mô hình dự báo phiên kế tiếp cho {sym}."
            " Bạn vui lòng thử lại sau."
        )
    return text
