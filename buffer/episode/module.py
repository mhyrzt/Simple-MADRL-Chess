import numpy as np


class Episode:
    def __init__(self) -> None:
        self.goals = []
        self.probs = []
        self.masks = []
        self.values = []
        self.states = []
        self.rewards = []
        self.actions = []

    def add(
        self,
        state: np.ndarray,
        reward: float,
        action,
        goal: bool,
        prob: float = None,
        value: float = None,
        masks: np.ndarray = None,
    ):
        self.goals.append(goal)
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

        if prob is not None:
            self.probs.append(prob)
        if value is not None:
            self.values.append(value)
        if masks is not None:
            self.masks.append(masks)

    def calc_advantage(self, gamma: float, gae_lambda: float) -> np.ndarray:
        n = len(self.rewards)
        advantages = np.zeros(n)
        for t in range(n - 1):
            discount = 1
            for k in range(t, n - 1):
                advantages[t] += (
                    discount
                    * (
                        self.rewards[k]
                        + gamma * self.values[k + 1] * (1 - int(self.goals[k]))
                    )
                    - self.values[k]
                )
                discount *= gamma * gae_lambda
        return list(advantages)

    def __len__(self):
        return len(self.goals)

    def total_reward(self) -> float:
        return sum(self.rewards)