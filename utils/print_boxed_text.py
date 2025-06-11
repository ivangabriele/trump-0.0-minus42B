import textwrap


def print_boxed_text(text: str, width: int, horizontal_line_char: str) -> None:
    wrapped_lines = textwrap.wrap(text, width=width - 4)

    for line in wrapped_lines:
        print(f"{horizontal_line_char} {line}" + " " * (width - len(line) - 3) + horizontal_line_char)
