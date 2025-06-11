from typing import Optional
import utils


def print_horizontal_line(line_char: str, title: Optional[str] = None) -> None:
    width = utils.get_terminal_width()

    if not title:
        print(line_char * width)

        return

    title_with_padding = f" {title} "
    title_length = len(title_with_padding)

    if title_length >= width:
        print(title_with_padding[:width])

        return

    side_len = (width - title_length) // 2
    extra = (width - title_length) % 2

    left = line_char * side_len
    right = line_char * (side_len + extra)

    print(f"{left}{title_with_padding}{right}")
