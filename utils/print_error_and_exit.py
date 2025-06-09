def print_error_and_exit(message: str, exception: Exception) -> None:
    print(f"Error: {message}")
    print(exception)
    exit(1)
