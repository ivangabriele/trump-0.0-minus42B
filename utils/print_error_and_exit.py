from typing import Optional


def print_error_and_exit(message: str, exception: Optional[Exception] = None) -> None:
    print(f"Error: {message}")
    if exception:
        print(exception)

    exit(1)
