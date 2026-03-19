from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .application import main as application_main

if TYPE_CHECKING:
    from collections.abc import Sequence
logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return application_main(argv)
    except Exception:
        logger.exception("uncaught exception", extra={"file_only": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
