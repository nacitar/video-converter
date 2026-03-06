import logging
from typing import Sequence

from .application import main as application_main

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return application_main(argv)
    except Exception as e:
        logger.exception(str(e), extra={"file_only": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
