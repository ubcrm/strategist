from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
from src.constants import SETTINGS, TORCH_DEVICE

from src.agent.entities.player import PlayerId, Player
from src.agent.entities.robot import RobotId, Robot
from src.util.point import Point
import torch


def pos_distance(from_pos: Point, to_pos: Point) -> float:
    return from_pos.distance_to(to_pos)


def pos_from_indices(indices: Tuple[int, int]):
    return Point(indices[0], indices[1])


class Board:
    def __init__(self, board: Dict[str, Any]):
        # Square board

        self._players = None
        self._robots = None

        self._populate_objs()

        self.settings = SETTINGS["board"]
        self._current_player_id = self.settings["player_id"]
        self.size = self.settings["size"]
        self.dims = tuple(self.settings["dims"])

        self._ordered_player_map = self.calculate_p_id_map()

        self.map = self.parse_map()
        self.additional_vals = self.get_additional_board_vals()

    def _populate_objs(self):
        pass

    @property
    def sorted_player_ids(self) -> List[int]:
        """
        The ordered player ID list. Used for parsing the board.
        :return: A list of all player IDs with the agent's player ID at index 0
        """
        return list(self._ordered_player_map.keys())

    def calculate_p_id_map(self) -> Dict[int, int]:
        ids = set(self.players.keys())
        ids.remove(self.current_player_id)
        id_list = [self.current_player_id] + list(ids)
        return {v: idx for idx, v in enumerate(id_list)}

    def parse_map(self) -> np.ndarray:
        pass

    def robot_map(self, player_id) -> np.ndarray:
        return self.map[player_id + 1, :, :]

    def get_additional_board_vals(self):
        pass

    def list_pos_to_board_pos(self, list_pos: int) -> Point:
        """
        Convert 1d list position to 2d board position
        :param list_pos: 1d position in list
        """
        p = divmod(list_pos, self.size)
        return Point(p[1], p[0])

    @property
    def players(self) -> Dict[PlayerId, 'Player']:
        return self._players

    @property
    def player(self) -> 'Player':
        return self._players[self.current_player_id]

    @property
    def robots(self) -> Dict[RobotId, 'Robot']:
        return self._robots


from src.agent.entities.robot import Robot, RobotId

if __name__ == "__main__":
    a = np.zeros((10, 10, 10))

    a[:, 1, 1] = [i for i in range(10)]
