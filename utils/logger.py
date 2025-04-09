from datetime import datetime, timezone, timedelta
import logging

def utc_to_utc8(utc_dt):
    return utc_dt.astimezone(timezone(timedelta(hours=8)))

# Override the default logging time formatter to use UTC+8
class UTC8Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        utc_dt = datetime.fromtimestamp(record.created, timezone.utc)
        utc8_dt = utc_to_utc8(utc_dt)
        if datefmt:
            return utc8_dt.strftime(datefmt)
        else:
            return utc8_dt.isoformat()