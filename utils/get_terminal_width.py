import os
import sys


def get_terminal_width() -> int:
    file_descriptor = sys.stdout.fileno()  # Should be `1` (stdout)
    terminal_size = os.get_terminal_size(file_descriptor)

    return terminal_size.columns
