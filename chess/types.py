import numpy as np

Cell = tuple[int]
Action = tuple[Cell, Cell]
Trajectory = tuple[np.ndarray, float, bool, dict]