"""MCP server for the Twitter screenshot archive."""

import sys


def main():
    try:
        from .server import main as _main
    except ImportError:
        print(
            "Error: MCP dependencies are not installed.\n"
            "Install the mcp extra:  uv sync --extra mcp",
            file=sys.stderr,
        )
        sys.exit(1)
    _main()
