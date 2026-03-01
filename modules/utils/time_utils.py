from __future__ import annotations

from datetime import datetime, timedelta
import pytz
from modules.core.state import GlobalState
from modules.api.time_api import get_now

VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

def resolve_time_window(
        state: GlobalState,
        default_hours: int = 72,
    ) -> tuple[int, int]:
    """
    Trả về (start_ts, end_ts) theo timezone VN
    - Nếu state.time_filter đã có (do Processor detect "hôm nay", "ngày cụ thể")
    -> dùng luôn cặp đó
    - Nếu chưa có -> mặc định lấy khoảng [now - default_hours, now].

    Giúp:
    - Processor chỉ cần set time_filter sau khi user nói rõ thời gian.
    - Các node (vector_db, retriever, API...) dùng chung 1 logic, không lệch nhau.
    """
    tf = getattr(state, "time_filter", None)
    if tf and isinstance(tf, (tuple, list)) and len(tf) == 2:
        start_ts, end_ts = tf
        return int(start_ts), int(end_ts)
    
    now = get_now()
    start_ts = int((now - timedelta(hours=default_hours)).timestamp())
    end_ts = int(now.timestamp())
    return start_ts, end_ts