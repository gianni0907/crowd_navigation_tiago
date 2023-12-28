from enum import Enum

class FSMStates(Enum):
    IDLE = 0
    START = 1
    ACTIVE = 2
    HOLD = 3