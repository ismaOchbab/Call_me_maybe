"""
Minimal CLI visualization of the constrained generation process
"""

GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"

_enabled = True


def set_enabled(value: bool) -> None:
    """Turn the visualization on or off (off by default)"""
    global _enabled
    _enabled = value


def start(label: str) -> None:
    """Announce the field about to be generated"""
    if _enabled:
        print(f"{YELLOW}{label}{RESET}")


def step(allowed: int, total: int, text: str, buffer: str) -> None:
    """Show one decoding step: mask size, token picked, value so far"""
    if _enabled:
        print(
            f"  {DIM}{allowed:>6}/{total}{RESET} "
            f"{GREEN}{text!r}{RESET} {DIM}-> {buffer}{RESET}"
        )


def done(value: object) -> None:
    """Show the resolved value of the field"""
    if _enabled:
        print(f"  {GREEN}= {value!r}{RESET}\n")
