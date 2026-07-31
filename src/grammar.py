def is_complete_number(text: str) -> bool:
    """
    Returns True if 'text' is fully valid, finished JSON number
    """
    if not text:
        return False
    index = 0
    if text[0] == '-':
        index = 1
    if index >= len(text):
        return False
    digits_before_dot = 0
    while index < len(text) and text[index].isdigit():
        digits_before_dot += 1
        index += 1
    if digits_before_dot == 0:
        return False
    if index == len(text):
        return True
    if text[index] != '.':
        return False
    index += 1
    digits_after_dot = 0
    while index < len(text) and text[index].isdigit():
        digits_after_dot += 1
        index += 1
    return digits_after_dot > 0 and index == len(text)


def is_valid_number_prefix(text: str) -> bool:
    """Return True if `text` could still become a valid JSON number
    with more characters appended (or already is one)"""
    if not text:
        return True
    index = 0
    if text[0] == "-":
        index = 1
    digits_before_dot = 0
    while index < len(text) and text[index].isdigit():
        digits_before_dot += 1
        index += 1
    if index == len(text):
        # Ended after only a "-" (0 digits) or after some digits — both
        # are fine as a prefix, e.g. "-" alone, or "3", or "42".
        return True
    if text[index] != ".":
        return False
    if digits_before_dot == 0:
        return False  # a dot needs at least one digit before it
    index += 1
    digits_after_dot = 0
    while index < len(text) and text[index].isdigit():
        digits_after_dot += 1
        index += 1

    return index == len(text)


class NumberConstraint:
    """
    Tracks progress through generating a valid JSON number,
    one caracter at a time
    """

    def __init__(self) -> None:
        self.buffer = ""

    def extends(self, text: str) -> bool:
        """
        """
        return is_valid_number_prefix(self.buffer + text)

    def append(self, text: str) -> None:
        """
        Adds 'text' to buffer. caller must have already confirmed extends(text)
        """
        self.buffer += text

    def resolved_value(self) -> float:
        """
        Returns the buffer as float
        """
        if not is_complete_number(self.buffer):
            raise ValueError(
                f"'{self.buffer}' is not a complete number"
            )
        return float(self.buffer)


class StringConstraint:
    """
    Tracks progress through generating the content of a JSON string
    """
    def __init__(self) -> None:
        self.buffer = ""

    def extends(self, text: str) -> bool:
        """
        Whether appending 'text' keeps this valid, ongoing string content
        """
        return '"' not in text

    def append(self, text: str) -> None:
        self.buffer += text

    def is_closing_quote(self, text: str) -> bool:
        return text == '"' and bool(self.buffer)

    def resolved_value(self) -> str:
        """
        Returns the finished string content
        """
        return self.buffer


class EnumConstraint:
    """
    Tracks the progress through generating text that must exactly match
    one of a fixed list of candidate strings
    """

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = list(candidates)
        self.consumed = "" # buffer or what's been typed so far

    def extends(self, text: str) -> bool:
        """
        Checks whether text is consistent with remaining candidates
        """
        candidate_prefix = self.consumed + text
        return any(c.startswith(candidate_prefix) for c in self.candidates)

    def append(self, text: str) -> None:
        """
        Commits text and narrow the candidate list to only those still
        consistent with what's been consumed
        """
        self.consumed += text
        self.candidates = [
            c for c in self.candidates if c.startswith(self.consumed)
            ]

    def is_resolved(self) -> bool:
        """
        checks whether one candidate remains and matches consumed exactly
        """
        return len(self.candidates) == 1 and\
            self.consumed == self.candidates[0]

    def resolved_value(self) -> str:
        """
        Returns the single candidate this generation resolved to
        """
        if not self.is_resolved():
            raise ValueError(
                f"Enum constraint not resolved: consumed={self.consumed}, "
                f"remaining={self.candidates!r}"
            )
        return self.candidates[0]


