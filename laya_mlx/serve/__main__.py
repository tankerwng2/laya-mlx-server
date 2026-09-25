# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""``python -m laya_mlx.serve {http,mcp}`` -- the two surfaces, one runtime."""

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m laya_mlx.serve", description=__doc__)
    sub = parser.add_subparsers(dest="surface", required=True)
    sub.add_parser("http", help="Jev-compatible HTTP server (POST /v1/systemone)")
    sub.add_parser("mcp", help="MCP stdio server")
    args = parser.parse_args(argv)
    if args.surface == "http":
        from .http import main as http_main

        http_main()
    else:
        from .mcp import main as mcp_main

        mcp_main()


if __name__ == "__main__":
    main()
