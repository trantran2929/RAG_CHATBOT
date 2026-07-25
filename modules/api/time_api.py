from datetime import datetime, timedelta
import calendar
import re
import pytz

WEEKDAYS_VN = {
    0: "Thứ hai", 1: "Thứ Ba", 2: "Thứ Tư", 3:"Thứ Năm",
    4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"
}

def get_now(tz: str = "Asia/Ho_Chi_Minh") -> datetime:
    return datetime.now(pytz.timezone(tz))

def format_full(dt: datetime) -> str:
    if dt is None:
        dt = get_now()
    weekday = WEEKDAYS_VN[dt.weekday()]
    return f"{weekday}, Ngày {dt.day:02d} tháng {dt.month:02d} năm {dt.year}"

def format_date(dt: datetime)->str:
    return f"Ngày {dt.day:02d} tháng {dt.month:02d} năm {dt.year}"

def format_weekday(dt: datetime) -> str:
    return WEEKDAYS_VN[dt.weekday()]

def add_time(now: datetime, num: int, unit: str) -> datetime:
    if unit == "ngày":
        return now + timedelta(days=num)
    elif unit == "tuần":
        return now + timedelta(weeks=num)
    elif unit == "tháng":
        month = now.month + num
        year = now.year + (month - 1)//12
        month = (month-1)%12 + 1
        day = min(now.day, calendar.monthrange(year, month)[1])
        return now.replace(year=year, month=month, day=day)
    elif unit == "năm":
        target_year = now.year + num
        day = min(now.day, calendar.monthrange(target_year, now.month)[1])
        return now.replace(year=target_year, day=day)
    return now


def normalize_time_query(query: str) -> str:
    """Chuẩn hóa viết tắt/lỗi phổ biến chỉ trong các cụm ngày giờ rõ ràng."""
    q = re.sub(r"\s+", " ", query or "").strip()
    q = re.sub(r"\bngay(?=\s+(?:mai|kia|\d{1,2}\b|may\b|bao\s+nhieu\b))", "ngày", q, flags=re.I)
    q = re.sub(r"\bnam(?=\s+\d{4}\b)", "năm", q, flags=re.I)
    q = re.sub(r"\bháng\b", "tháng", q, flags=re.I)
    q = re.sub(
        r"(\bngày\s+\d{1,2}\s+)(?:t|th|thg)(?=\s+\d{1,2}\b)",
        r"\1tháng",
        q,
        flags=re.I,
    )
    q = re.sub(r"\b(?:là|la|lag)\s+(?:th|thu)\s+(?:may|mấy)\b", "là thứ mấy", q, flags=re.I)
    q = re.sub(r"\b(?:là|la|lag)\s+(?:ngày|ngay)\s+(?:may|mấy)\b", "là ngày mấy", q, flags=re.I)
    q = re.sub(r"\b(?:la|lag)(?=\s+(?:thứ|ngày)\b)", "là", q, flags=re.I)
    return q


def answer_time_query(query: str, now: datetime | None = None) -> str:
    """Answer common Vietnamese date/time questions in Viet Nam timezone."""
    now = now or get_now()
    q = normalize_time_query(query).lower()

    if "còn bao lâu" in q and ("cuối năm" in q or "hết năm" in q):
        end = now.replace(month=12, day=31, hour=23, minute=59, second=59)
        diff = max(end - now, timedelta(0))
        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes = remainder // 60
        return f"Còn khoảng {days} ngày {hours} giờ {minutes} phút nữa là hết năm."

    target = now
    target_label = "Hôm nay"
    if "hôm qua" in q:
        target = add_time(now, -1, "ngày")
        target_label = "Hôm qua"
    elif "ngày kia" in q:
        target = add_time(now, 2, "ngày")
        target_label = "Ngày kia"
    elif "ngày mai" in q or re.search(r"\bmai\b", q):
        target = add_time(now, 1, "ngày")
        target_label = "Ngày mai"
    else:
        relative = re.search(r"(\d+)\s+ngày\s+(?:nữa|sau|tới)", q)
        if relative:
            amount = int(relative.group(1))
            target = add_time(now, amount, "ngày")
            target_label = f"{amount} ngày nữa"

    absolute = re.search(r"(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", q)
    if absolute is None:
        absolute = re.search(
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+năm\s+(\d{4}))?",
            q,
        )
    looks_like_explicit_date = bool(
        re.search(r"\bngày\s+\d{1,2}\b", q)
        and (re.search(r"\b(?:tháng|thg|t|năm)\b", q) or re.search(r"[/-]", q))
    )
    if absolute is None and looks_like_explicit_date:
        return "Mình không nhận diện được ngày bạn nhập. Vui lòng dùng dạng dd/mm/yyyy."
    if absolute:
        day, month, year = absolute.groups()
        try:
            target = now.replace(
                year=int(year or now.year),
                month=int(month),
                day=int(day),
            )
            target_label = f"Ngày {int(day):02d}/{int(month):02d}/{int(year or now.year)}"
        except ValueError:
            return "Ngày bạn nhập không hợp lệ."

    asks_relative_date = target.date() != now.date() or absolute is not None
    if "thứ mấy" in q or "thứ gì" in q:
        if absolute is not None:
            return f"{target_label} là {format_weekday(target)}."
        return f"{target_label} là {format_weekday(target)}, {target.strftime('%d/%m/%Y')}."
    if "ngày mấy" in q or "ngày bao nhiêu" in q:
        return f"{target_label} là ngày {target.strftime('%d/%m/%Y')}."
    if (
        "ngày nào" in q
        or asks_relative_date
    ):
        return f"{target_label} là {format_full(target)}."
    if "mấy giờ" in q or "giờ hiện tại" in q or "bây giờ" in q:
        return f"Hiện tại là {now.strftime('%H:%M:%S')}, ngày {now.strftime('%d/%m/%Y')}."
    return f"Hiện tại là {now.strftime('%H:%M:%S')}, {format_full(now)}."

def get_datetime_context()->str:
    now = get_now()
    return (
        f"Hôm nay là {format_full(now)}.\n"
        # f"Ngày hôm qua là {format_date(add_time(now, -1, 'ngày'))}.\n"
        # f"Ngày mai là {format_date(add_time(now, 1, 'ngày'))}.\n"
        # f"Tuần sau sẽ bắt đầu từ {format_date(add_time(now, 7, 'ngày'))}.\n"
        # f"Tuần trước bắt đầu từ {format_date(add_time(now, -7, 'ngày'))}."
    )

def get_current_time() -> str:
    """Trả về giờ hiện tại (HH:MM:SS)"""
    now = get_now()
    return now.strftime("%H:%M:%S")

def get_current_date() -> str:
    """Trả về ngày hiện tại (dd/mm/yyyy)"""
    now = get_now()
    return now.strftime("%d/%m/%Y")
