"""Run the existing src.run CLI with Hydra help compatibility for Python 3.14."""

import argparse
import runpy
import sys
from unittest.mock import patch


def main():
    if sys.version_info < (3, 14):
        runpy.run_module("src.run", run_name="__main__")
        return
    original = argparse.HelpFormatter._expand_help

    def expand_help(formatter, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return original(formatter, action)

    with patch.object(argparse.HelpFormatter, "_expand_help", expand_help):
        runpy.run_module("src.run", run_name="__main__")


if __name__ == "__main__":
    main()
