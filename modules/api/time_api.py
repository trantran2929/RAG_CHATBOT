from datetime import datetime, timedelta
import calendar
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
    return f"{weekday}, Ngày {dt.day:02d} tháng {dt.month:02d} năm {dt.year}, {dt.strftime('%H:%M:%S')}"
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
        return now.replace(year=now.year + num)
    return now
def get_datetime_context()->str:
    now = get_now()
    return (
        f"Hôm nay là {format_full(now)}.\n"
        f"Ngày hôm qua là {format_date(add_time(now, -1, 'ngày'))}.\n"
        f"Ngày mai là {format_date(add_time(now, 1, 'ngày'))}.\n"
        f"Tuần sau sẽ bắt đầu từ {format_date(add_time(now, 7, 'ngày'))}.\n"
        f"Tuần trước bắt đầu từ {format_date(add_time(now, -7, 'ngày'))}."
    )

def get_current_time() -> str:
    """Trả về giờ hiện tại (HH:MM:SS)"""
    now = get_now()
    return now.strftime("%H:%M:%S")


def get_current_date() -> str:
    """Trả về ngày hiện tại (dd/mm/yyyy)"""
    now = get_now()
    return now.strftime("%d/%m/%Y")