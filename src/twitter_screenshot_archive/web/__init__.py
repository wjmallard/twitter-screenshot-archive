"""Flask web UI for the Twitter screenshot archive."""

import sys


def main():
    try:
        from .app import main as _main
    except ImportError:
        print(
            "Error: Flask is not installed.\n"
            "Install the web extra:  uv sync --extra web",
            file=sys.stderr,
        )
        sys.exit(1)
    _main()
