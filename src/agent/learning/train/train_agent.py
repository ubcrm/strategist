import math
import random

import numpy as np
import torch
from abc import ABCMeta
from typing import Any, Dict
from src.agent.agent import Agent
from src.constants import TORCH_DEVICE, SETTINGS
from src.agent.learning.robot_agent import RobotAgent
from src.agent.learning.train.evaluator import evaluate_board
from src.parser.parser import parse_robot_input



class TrainAgent(Agent):
    def __init__(self, observation: Dict[str, Any], robot_mem, n_steps):
        super().__init__(observation)
        self.robot_memory = robot_mem
        self.step_no = n_steps

    def act(self) -> Dict[str, str]:
        robot_agent = TrainRobotAgent(self.robot_memory).to(TORCH_DEVICE)

        e_end = SETTINGS["learn"]["eps_end"]
        e_start = SETTINGS["learn"]["eps_start"]
        e_decay = SETTINGS["learn"]["eps_decay"]
        eps_threshold = e_end + (e_start - e_end) * math.exp(-1. * self.step_no / e_decay)

        for robot in self.robots:
            sample = random.random()
            if sample > eps_threshold:
                s_action = robot_agent.act(robot, self.board)
                robot.next_action = s_action
            else:
                robot.next_action = (random.uniform(-1, 1), random.uniform(-1, 1))
        return self.get_next_actions()


class TrainRobotAgent(RobotAgent, metaclass=ABCMeta):
    def __init__(self, memory):
        super(RobotAgent, self).__init__()
        self.memory = memory

    def act(self, robot: "Robot", board: "Board"):
        robot_input = parse_robot_input(robot, board)
        forward_res = self.forward(robot_input)

        action = torch.tensor([forward_res.argmax().item()], device=TORCH_DEVICE)
        value = torch.tensor([evaluate_board(board)], device=TORCH_DEVICE)
        self.memory.cache_state(robot.id, board.step, robot_input, action, value)
        return action.argmax().item()


from src.agent.entities.robot import Robot
