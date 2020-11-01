from typing import Dict, Any

from src.constants import TORCH_DEVICE, SETTINGS
from src.agent.board.board import Board


class Agent:
    def __init__(self, observation: Dict[str, Any]):
        self.observation = observation

        self.robot_states = {}
        self.board = Board(observation)
        self.robots = []

    def act(self) -> Dict[str, str]:
        return self.get_next_actions()

    def get_next_actions(self) -> Dict[str, str]:
        pass

    def get_robot_states(self):
        pass