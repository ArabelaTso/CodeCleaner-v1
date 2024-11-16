import structlog
import logging


def config_logger() -> structlog.BoundLogger:
    # console processors
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.add_log_level,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    std_logger = logging.getLogger()
    std_logger.handlers.clear()  # remove default handler
    std_logger.addHandler(console_handler)
    std_logger.setLevel("INFO")

    structlog.configure(
        processors=[
            # Prepare event dict for `ProcessorFormatter`.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    return structlog.get_logger()
