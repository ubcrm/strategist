# Imports helper functions
from enum import Enum, auto
from typing import Optional, NewType, Tuple

from src.agent.board.board import pos_distance

# Directions a robot can move
from src.agent.entities.player import PlayerId
from src.util.point import Point

RobotId = NewType('RobotId', str)


class RobotState(Enum):
    IDLE = auto()
    MOVE = auto()


class Robot:
    """
    Agent for an individual robot. Contains things like pathfinding to desired location,
    and performing robot actions
    """

    def __init__(
            self,
            robot_id: RobotId,
            position: Point,
            player_id: PlayerId,
    ):
        super().__init__(robot_id, position, player_id)
        self.state = RobotState.COLLECT
        self.position = position
        self.controller = player_id

    def move_to(self, loc: Point):
        self.next_action = self.get_dir_to(loc)

    # Pathfinds to location
    def get_dir_to(self, to_pos: Point, board: "Board") -> Optional[Tuple[int, int]]:
        pass


from src.agent.board.board import Board
