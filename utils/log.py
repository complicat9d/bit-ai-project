import logging


def create_logger():
    _logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s:%(lineno)i - %(message)s"
    )

    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

    return _logger


logger = create_logger()
