from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class DecodeState(Enum):
    """
    Possible states of the constrained decoder
    """
    START = auto()
    FUNCTION_NAME = auto()
    PARAMETERS_START = auto()
    PARAMETER_NAME = auto()
    PARAMETER_VALUE = auto()
    PARAMETERS_END = auto()
    END = auto()


@dataclass
class DecoderContext:
    """
    Track the state of the constrained decoding
    """
    state: DecodeState = DecodeState.START
    function_name: Optional[str] = None
    parameter_index: int = 0