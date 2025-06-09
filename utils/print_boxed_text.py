import textwrap


def print_boxed_text(text: str, width: int) -> None:
    wrapped_lines = textwrap.wrap(text, width=width - 4)

    for line in wrapped_lines:
        print(f"┃ {line}" + " " * (width - len(line) - 3) + "┃")
