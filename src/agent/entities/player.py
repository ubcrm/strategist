from typing import List, NewType

from src.agent.entities.robot import RobotId

PlayerId = NewType('PlayerId', str)


class Player:
    def __init__(self, player_id: PlayerId, robot_ids: List[RobotId]):
        self._id = player_id
        self._robot_ids = robot_ids

    @property
    def robot_ids(self) -> List["RobotId"]:
        return self._robot_ids

    def get_robots(self, board: "Board"):
        pass

    from src.agent.board.board import Board
    from src.agent.entities.robot import Robot
