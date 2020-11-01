import math


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, p):
        return math.sqrt((p.x - self.x) ** 2 + (p.y - - self.y) ** 2)
