from datetime import datetime, timezone, timedelta
import logging


def utc_to_utc8(utc_dt):
    return utc_dt.astimezone(timezone(timedelta(hours=8)))


class UTC8Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        utc_dt = datetime.fromtimestamp(record.created, timezone.utc)
        utc8_dt = utc_to_utc8(utc_dt)
        if datefmt:
            return utc8_dt.strftime(datefmt)
        else:
            return utc8_dt.isoformat()


# class HTTPRequestFilter(logging.Filter):
#     def filter(self, record):
#         # Exclude log messages that contain "HTTP/1.1 200 OK"
#         return "HTTP/1.1 200 OK" not in record.getMessage()


# NEW FUNCTION ADDED
def setup_logging(log_file):
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = UTC8Formatter(fmt="%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # logger.addFilter(HTTPRequestFilter())
    return logger
