import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs"):
    # 建立 logs 資料夾
    os.makedirs(log_dir, exist_ok=True)

    # 取得當前時間作為檔名
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")

    # 設定格式
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 建立 formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 建立 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f"Logging initialized. Log file: {log_filename}")