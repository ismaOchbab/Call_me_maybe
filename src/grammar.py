"""
Grammar constraints for constrained decoding.

This module contains classes that enforce specific grammar rules during
token-by-token generation.

The three constraints are:
    - EnumConstraint:   force the output to exactly match one of a fixed
                        set of candidate strings (e.g., function names
                        or 'true'/'false').
    - NumberConstraint: force the output to be a valid JSON number
                        (e.g., 42, -3.14). The constraint
                        allows only tokens that keep the prefix valid.
    - StringConstraint: force the output to be the content of a JSON
                        string (without the surrounding quotes). The
                        constraint allows any token that does not
                        contain an unescaped double quote, and it
                        permits the closing quote at any time so the
                        model can decide when to stop.
"""


def is_complete_number(text: str) -> bool:
    """
    Check whether the given string is a fully valid, finished JSON number.

    Args:
        text: The string to check (e.g., "42", "-3.14").

    Returns:
        True if the string is a complete JSON number, False otherwise.
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
    """
    Check whether the string could be a prefix of a valid JSON number.

    Args:
        text: The partial string to test.

    Returns:
        True if appending more characters could eventually yield a
        complete JSON number, False if the prefix is already invalid.
    """
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
    Tracks progress through generating a valid JSON number, token by token

    The buffer holds the concatenated text of all tokens generated so far.
    The constraint ensures that the buffer always remains a valid prefix
    of a JSON number. When generation is complete, the buffer can be
    converted to a float.
    """

    def __init__(self) -> None:
        """Initialize an empty buffer."""
        self.buffer = ""

    def extends(self, text: str) -> bool:
        """
        Check whether appending the given text to the current buffer
        would keep it as a valid prefix of a JSON number.

        Args:
            text: The candidate token string to test.

        Returns:
            True if the concatenation is a valid number prefix,
            False otherwise.
        """
        return is_valid_number_prefix(self.buffer + text)

    def append(self, text: str) -> None:
        """
        Append a token to the buffer.

        This method does not perform validation; the caller must have
        already called extends() and confirmed the token is allowed.

        Args:
            text: The token string to append.
        """
        self.buffer += text

    def resolved_value(self) -> float:
        """
        Convert the buffer to a float.

        Raises ValueError if the buffer is not a complete number.

        Returns:
            The numeric value as a Python float.
        """
        if not is_complete_number(self.buffer):
            raise ValueError(
                f"'{self.buffer}' is not a complete number"
            )
        return float(self.buffer)


class StringConstraint:
    """
    Tracks progress through generating the content of a JSON string

    This constraint is applied to the content of the string.

    The model is free to output the closing quote at any time
    to signal that the string is complete.
    """
    def __init__(self) -> None:
        """Initialize an empty buffer."""
        self.buffer = ""

    def extends(self, text: str) -> bool:
        """
        Check whether appending the given token would keep the string
        content valid.

        A token is valid content if it does not contain a double quote
        character. This avoids the need for escaping; the model can
        only output raw text without quotes.

        Args:
            text: The candidate token string.

        Returns:
            True if the token contains no double quote, False otherwise.
        """
        return '"' not in text

    def append(self, text: str) -> None:
        """
        Append a token to the content buffer.

        Caller must have already validated the token with extends().
        """
        self.buffer += text

    def is_closing_quote(self, text: str) -> bool:
        """
        Determine if the given token is the closing double quote.

        Args:
            text: The token string to test.

        Returns:
            True if the token is exactly the double quote character,
            False otherwise.
        """
        return text == '"'

    def resolved_value(self) -> str:
        """
        Return the final string content.

        The buffer contains everything except the surrounding quotes.
        It is returned as-is (no further validation).

        Returns:
            The generated string content.
        """
        return self.buffer


class EnumConstraint:
    """
    Tracks progress through generating text that must exactly match
    one of a fixed list of candidate strings.

    The constraint works by narrowing down the list of candidates as
    tokens are appended. At each step, only tokens that keep at least
    one candidate consistent are allowed. The generation stops when
    exactly one candidate remains and the consumed text equals that
    candidate.

    This is useful for generating function names, boolean literals, or
    any fixed-vocabulary choice.
    """

    def __init__(self, candidates: list[str]) -> None:
        """
        Initialize with the list of allowed strings.

        Args:
            candidates: List of exact strings the generation must match.
                        Must be non-empty.
        """
        self.candidates = list(candidates)
        self.consumed = ""  # buffer or what's been typed so far

    def extends(self, text: str) -> bool:
        """
        Check whether appending the given token could still lead to
        a match with at least one remaining candidate

        args:
            text: The candidate token string

        Rturns:
            True if at least one candidate starts with consumed + text
            False otherwise.
        """
        candidate_prefix = self.consumed + text
        return any(c.startswith(candidate_prefix) for c in self.candidates)

    def append(self, text: str) -> None:
        """
        Commits text and reduces the candidate list to only those still
        consistent with what's been consumed
        """
        self.consumed += text
        self.candidates = [
            c for c in self.candidates if c.startswith(self.consumed)
            ]

    def is_resolved(self) -> bool:
        """
        Check whether the constraint has fully resolved to a single
        candidate that exactly matches the consumed text

        Return:
            True if exactly one candidate remains and it equals consumed,
            False otherwise
        """
        return len(self.candidates) == 1 and\
            self.consumed == self.candidates[0]

    def resolved_value(self) -> str:
        """
        Returns the single candidate this generation resolved to

        Raises ValueError if the constraint is not yet resolved
        """
        if not self.is_resolved():
            raise ValueError(
                f"Enum constraint not resolved: consumed={self.consumed}, "
                f"remaining={self.candidates!r}"
            )
        return self.candidates[0]
