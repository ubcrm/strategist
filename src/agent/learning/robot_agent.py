import base64
from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import SETTINGS, TORCH_DEVICE, ROBOT_AGENT_STATE_DICT

from src.agent.entities.robot import RobotId, Robot, RobotState
from src.parser.parser import parse_robot_input


class RobotAgent(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(RobotAgent, self).__init__()
        self.input_channels = SETTINGS["board"]["dims"][0]
        self.add_vals_size = SETTINGS["learn"]["num_additional_vals"]
        output_size = 6
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(self.input_channels, self.input_channels * 2, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(77, 200)
        self.fc2 = nn.Linear(200, output_size)

    def forward(self, board_input):
        board_input_dims = SETTINGS["board"]["dims"]

        def feed_forward_input(fwd_input):
            additional_vals, x = fwd_input.split([self.add_vals_size, fwd_input.size(0) - self.add_vals_size])
            additional_vals = additional_vals.view(1, -1)
            x = x.view(1, self.input_channels, *self.vision_dims)

            in_size = x.size(0)
            x = self.conv1(x)
            x = F.relu(self.mp1(x))
            x = self.conv2(x)
            x = F.relu(self.mp2(x))
            x = F.relu(self.conv_drop(x))
            x = x.view(in_size, -1)

            x = torch.cat((x, additional_vals), dim=1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            return x

        desired_input_sz = int(np.prod(board_input_dims) + self.add_vals_size)
        if board_input.size(0) == desired_input_sz:
            return feed_forward_input(board_input)
        out_tensors = []
        for state in board_input.split(desired_input_sz):
            out_tensors.append(torch.flatten(feed_forward_input(state)))
        return torch.stack(out_tensors)

    def act(self, robot: Robot, board: "Board"):
        robot_input = parse_robot_input(robot, board)
        return self.forward(robot_input).argmax().item()

    def copy(self):
        agent_copy = RobotAgent().to(TORCH_DEVICE)
        agent_copy.load_state_dict(self.state_dict())
        return agent_copy

    def load_recent_model(self):
        self.load_state_dict(ROBOT_AGENT_STATE_DICT)

    def load_base64(self, base64_str):
        # Write to temp file for kaggle submission
        with open("model.dat", "wb") as f:
            f.write(base64.b64decode(base64_str))
        self.load_state_dict(torch.load('model.dat'))


from src.agent.board.board import Board
