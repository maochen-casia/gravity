import logging
import os
import time
from datetime import datetime

class Logger:
    def __init__(self, log_dir, log_level=logging.INFO):
        """
        初始化日志记录器
        :param log_dir: 日志文件存储路径
        :param log_level: 日志等级（默认INFO）
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 生成带时间戳的日志文件名
        log_file = os.path.join(log_dir, f"log.txt")

        # 创建日志记录器
        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel(log_level)

        # 统一日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 文件处理器（记录所有级别日志）
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器（仅显示INFO及以上级别）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_metrics(self, epoch, metrics_dict):
        """
        记录训练指标
        :param epoch: 当前epoch数
        :param metrics_dict: 包含指标的字典，例如：
            {"train_loss": 0.56, "val_acc": 0.82}
        """
        log_str = f"Epoch {epoch:03d}"
        for key, value in metrics_dict.items():
            log_str += f" | {key}: {value:.4f}"
        self.info(log_str)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)