import sys
import logging
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO") -> None:
    """Инициализирует структурированное логирование для всего приложения."""
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Очищаем корневой логгер (на случай повторного вызова)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Создаём JSON-форматтер
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        json_ensure_ascii=False
    )

    # Обработчик: вывод в stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(json_formatter)

    root_logger.addHandler(handler)

    # Отдельный логгер для агента
    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(log_level)

    # Отключаем распространение в корень (чтобы не дублировать)
    agent_logger.propagate = False
    agent_logger.addHandler(handler)
