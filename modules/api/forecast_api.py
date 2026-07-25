from typing import Dict, Any, Optional

from modules.ML.pipeline import smart_predict, predict_next_session


def _safe_get(d: Dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


CONFIDENCE_LABELS = {
    "high": "cao",
    "medium": "trung bình",
    "low": "thấp",
    "uncertain": "chưa chắc chắn",
    "up_confident": "khá chắc chắn theo hướng tăng",
    "down_confident": "khá chắc chắn theo hướng giảm",
}


def _confidence_label(value: Any) -> str:
    raw = str(value or "uncertain").strip()
    return CONFIDENCE_LABELS.get(raw, raw)


def _valid_band(band: Any) -> Optional[Dict[str, float]]:
    if not isinstance(band, dict):
        return None
    try:
        mean = float(band["px_mean"])
        low = float(band["px_lo"])
        high = float(band["px_hi"])
    except (KeyError, TypeError, ValueError):
        return None
    if low > high:
        low, high = high, low
    return {"px_mean": mean, "px_lo": low, "px_hi": high}


def _model_note(pack: Dict[str, Any]) -> str:
    gap = pack.get("gap") if isinstance(pack.get("gap"), dict) else {}
    trained_through = gap.get("model_trained_through")
    model = pack.get("model")
    source = pack.get("base_from") or pack.get("source_used")
    parts = []
    if trained_through:
        parts.append(f"model học tới phiên {trained_through}")
    if model:
        parts.append(str(model))
    if source == "daily_fallback":
        parts.append("fallback dữ liệu ngày do thiếu intraday")
    elif source == "last_close_daily":
        parts.append("dùng dữ liệu ngày do thiếu intraday buổi sáng")
    return f"- Nguồn mô hình: {', '.join(parts)}." if parts else ""


def _format_intraday_pack(sym: str, pack: Dict[str, Any]) -> str:
    if not isinstance(pack, dict) or pack.get("mode") != "in_session":
        return ""
    if pack.get("next_step_dir") is None:
        detail = pack.get("error") or "không đủ dữ liệu đầu vào"
        return f"⚠️ Chưa thể dự báo nội phiên cho **{sym}**: {detail}"

    confidence = _confidence_label(pack.get("step_confidence"))
    lines = [
        f"📈 Dự báo ngắn hạn nội phiên cho **{sym}** ({pack.get('session', 'AM/PM')}):",
        f"- Xu hướng kế tiếp: **{pack['next_step_dir']}** (độ tin cậy {confidence}).",
    ]
    last_px = pack.get("last_px")
    if isinstance(last_px, (int, float)) and last_px > 0:
        lines.append(f"- Giá gần nhất: {float(last_px):,.0f} VNĐ.")
    path = pack.get("path_pred")
    if path is not None and hasattr(path, "tolist"):
        values = [float(v) for v in path.tolist() if v is not None]
        if values:
            lines.append(
                "- Quỹ đạo 3 bước tham khảo: "
                + ", ".join(f"{value:,.0f} VNĐ" for value in values[:3]) + "."
            )
    note = _model_note(pack)
    if note:
        lines.append(note)
    if pack.get("error"):
        lines.append(f"- Lưu ý dữ liệu: {pack['error']}")
    lines.append("⚠️ Đây là ước lượng thống kê, không phải khuyến nghị mua/bán.")
    return "\n".join(lines)


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
            close_dir = pack.get("close_direction", pack.get("open_direction"))
            close_return_pct = pack.get(
                "close_return_pct", pack.get("open_gap_pct")
            )
            if close_dir is not None and close_return_pct is not None:
                return (
                    f"🔮 Phiên kế tiếp của {sym} ({target_day}): giá đóng cửa "
                    f"có khả năng **{close_dir}** khoảng "
                    f"{close_return_pct:+.2f}% (tham khảo)."
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
            band = _valid_band(pack.get("close_band", pack.get("open_band")))
            if band is None:
                return ""
            px_mean, px_lo, px_hi = band["px_mean"], band["px_lo"], band["px_hi"]
            direction = pack.get(
                "close_direction", pack.get("open_direction", "dao động nhẹ")
            )
            gap_pct = pack.get("close_return_pct", pack.get("open_gap_pct", 0.0))
            conf = _confidence_label(
                pack.get("close_confidence", pack.get("open_confidence"))
            )
            model_note = _model_note(pack)
            signal = pack.get("trade_signal", "NO_TRADE")

            msg = [
                f"📅 Phiên giao dịch kế tiếp của {sym} ({target_day}):",
                f"- Giá đóng cửa dự kiến khoảng {px_mean:,.2f} VNĐ "
                f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
                f"- Dự kiến {direction} khoảng {gap_pct:+.2f}%.",
                f"- Tín hiệu sau ngưỡng chi phí: **{signal}**.",
                f"- Mức độ tự tin mô hình: {conf}.",
                *([model_note] if model_note else []),
                "\n",
                "⚠️ Đây chỉ là ước lượng dựa trên tin tức & hành vi giá gần nhất,"
                " không phải khuyến nghị đầu tư."
            ]
            return "\n".join(msg)

        if next_sess == "PM":
            band = _valid_band(pack.get("pm_band"))
            if band is None:
                return ""
            px_mean, px_lo, px_hi = band["px_mean"], band["px_lo"], band["px_hi"]
            direction = pack.get("pm_direction", "dao động nhẹ")
            gap_pct = pack.get("pm_gap_pct", 0.0)
            conf = _confidence_label(pack.get("pm_confidence"))
            model_note = _model_note(pack)

            msg = [
                f"📅 Phiên chiều (PM) kế tiếp của {sym} ({target_day}):",
                f"- Giá tham chiếu đầu phiên chiều dự kiến quanh {px_mean:,.2f} VNĐ "
                f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
                f"- Khuynh hướng {direction} khoảng {gap_pct:+.2f}%.",
                f"- Độ tin cậy: {conf}.",
                *([model_note] if model_note else []),
                "\n ⚠️ Đây là thông tin tham khảo, không phải lời khuyên giao dịch."
            ]
            return "\n".join(msg)

    # 2) Gói out_of_session (predict_tomorrow_full_exog)
    if mode == "out_of_session":
        bands = pack.get("bands", {}) or {}
        close_band = _valid_band(bands.get("NEXT_CLOSE", bands.get("OPEN_am")))
        if close_band is None:
            return ""
        px_mean, px_lo, px_hi = (
            close_band["px_mean"],
            close_band["px_lo"],
            close_band["px_hi"],
        )
        close_dir = pack.get(
            "close_direction", pack.get("open_direction", "dao động nhẹ")
        )
        close_return_pct = pack.get(
            "close_return_pct", pack.get("open_gap_pct", 0.0)
        )
        conf = _confidence_label(
            pack.get("close_confidence", pack.get("open_confidence"))
        )
        target_day = pack.get("target_day")
        signal = pack.get("trade_signal", "NO_TRADE")

        msg = [
            f"📅 Phiên tiếp theo của {sym} ({target_day}):",
            f"- Giá đóng cửa dự kiến quanh {px_mean:,.2f} VNĐ "
            f"(dải {px_lo:,.2f} ~ {px_hi:,.2f}).",
            f"- Xu hướng khả năng {close_dir} khoảng {close_return_pct:+.2f}%",
            "  so với giá đóng cửa gần nhất.",
            f"- Tín hiệu sau ngưỡng chi phí: **{signal}**.",
            f"- Mức tự tin mô hình: {conf}.",
            "\n ⚠️ Đây chỉ là mô phỏng thống kê, KHÔNG phải khuyến nghị mua/bán."
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
    if not isinstance(pack, dict):
        return f"⚠️ Chưa có dữ liệu dự báo hợp lệ cho **{sym}**."
    mode = pack.get("mode", "")

    # 1. ĐANG TRONG PHIÊN → bước kế tiếp + phiên kế tiếp
    if mode == "in_session":
        intraday_text = _format_intraday_pack(sym, pack)

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
    open_band = _valid_band((pack.get("bands", {}) or {}).get("OPEN_am"))
    if open_band is None:
        return f"⚠️ Gói dự báo của **{sym}** thiếu dải giá hợp lệ."
    px_mean, px_lo, px_hi = open_band["px_mean"], open_band["px_lo"], open_band["px_hi"]
    open_dir = pack.get("open_direction", "dao động nhẹ")
    open_gap_pct = pack.get("open_gap_pct", 0.0)
    conf = _confidence_label(pack.get("open_confidence"))
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

    # Chỉ gọi lại predictor khi smart_predict thực sự lỗi/không trả schema hợp lệ.
    # Pack next_session/out_of_session đã đủ dữ liệu để format, không gọi API hai lần.
    if not isinstance(main_pack, dict) or main_pack.get("mode") not in (
        "in_session", "next_session", "out_of_session"
    ):
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

    return _format_intraday_pack(sym, pack)


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
