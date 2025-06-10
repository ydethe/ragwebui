"""

.. include:: ../../README.md

# CLI usage

ragwebui comes with a CLI tool called ragwebui.

# Testing

## Run the tests

To run tests, just run:

    uv run pytest

## Test reports

[See test report](../tests/report.html)

[See coverage](../coverage/index.html)

.. include:: ../../CHANGELOG.md

"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pythonjsonlogger import jsonlogger

from .config import config


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("ragwebui_logger")
logger.setLevel(config.LOGLEVEL.upper())

# Create stream handler for stdout
logHandler = logging.StreamHandler(sys.stdout)

# JSON formatter
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
