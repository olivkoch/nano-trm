import logging
import os


def setup_logging(log_level: str, log_filename: str | None) -> None:
    """
    Setup logging library
    Args:
        log_level: log level for root logger, as a string. Eg: "INFO"
        log_filename: If not None, will also log to disk, to the specified filename
    """
    # set log level
    log_level = logging.getLevelName(log_level)

    message_format = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        force=True,
        format=message_format,
        datefmt=date_format,
    )

    # add file handler if requested
    if log_filename:
        logging.info("Also logging to %s", log_filename)
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        file_handler = logging.FileHandler(filename=log_filename)
        file_handler.setFormatter(fmt=logging.Formatter(fmt=message_format, datefmt=date_format))
        logging.getLogger().addHandler(file_handler)
