import os
import logging

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple

from rl_tictactoe.policy_iteration import PolicyIteration
from rl_tictactoe.tictactoe import rng

DEBUG_STATE = 81
DEBUG_ACTION = 4
# Setting up the logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

file_handler = logging.FileHandler(f"{logs_dir}/q_learning.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

from rich.progress import track


class QLearning(PolicyIteration):
    """Q Learning algorithm for Tic Tac Toe"""
    def __init__(
        self,
        board_size: int = 3,
        win_condition: int = 3,
        gamma=np.float16(0.9),
        learning_rate=np.float16(0.1),
    ) -> None:
        super().__init__(board_size, win_condition, gamma)
        self.q_table = (
            rng.random(
                (self.num_states, self.board_size**2),
            ).astype(np.float16)
            * 1e-3
        )
        self.learning_rate = learning_rate

        self.valid_states = np.zeros(self.num_states, dtype=bool)
        self.index_to_state_map: Dict[int, npt.NDArray[np.int8]] = {}
        self.available_actions_cache: Dict[int, List[int]] = {}
        self.rewards = np.zeros(self.num_states, dtype=np.float16)
        self.state_action_to_next_state_map = (
            np.ones((self.num_states, self.num_cells), dtype=int) * -1
        )

        for i in np.arange(self.num_states, dtype=int):
            self.index_to_state_map[i] = self.index_to_state(i)
            state = self.index_to_state_map[i]
            self.valid_states[i] = self.is_valid_state(state)
            if self.valid_states[i]:
                self.available_actions_cache[i] = [
                    self.action_to_int(i) for i in self.available_actions(state)
                ]
                self.rewards[i] = self.reward_function(state)

        for i in np.arange(self.num_states, dtype=int):
            state = self.index_to_state_map[i]
            if self.valid_states[i]:
                available_actions = self.available_actions(state)
                for action in available_actions:
                    next_state = state.copy()
                    next_state[action[0]][action[1]] = self.which_players_turn(state)
                    next_state = tuple(next_state.flatten())
                    next_state = self.state_to_index_map[next_state]
                    self.state_next_state_to_action[i][next_state] = action
                    action = self.action_to_int(action)
                    self.state_action_to_next_state_map[i, action] = next_state

    def get_policy(
        self,
    ) -> Tuple[npt.NDArray[np.int8], npt.NDArray[np.float16]]:
        """
        Returns:
            npt.NDArray[np.int8]: policy based on the q table
            npt.NDArray[np.float16]: the q table
        """
        self.q_learning()
        players_turn = np.array(
            [self.which_players_turn(i) for i in self.index_to_state_map.values()],
            dtype=np.int8,
        )

        invalid_actions_mask = np.ones((self.num_states, self.board_size**2), dtype=bool) * np.inf

        for i in np.arange(self.num_states, dtype=int):
            if self.valid_states[i]:
                invalid_actions_mask[i, self.available_actions_cache[i]] = False

        self.q_table += invalid_actions_mask

        policy_ints = np.argmax(self.q_table * players_turn[:, np.newaxis], axis=1)
        policy = np.ones((self.num_states, 2), dtype=int) * -1
        for i in np.arange(self.num_states, dtype=int):
            if self.valid_states[i]:
                policy[i] = self.int_to_action(policy_ints[i])
        return policy, self.q_table

    def update_q_table(
        self,
        state: int,
        action: int,
    ) -> None:
        """Updates the q table

        Args:
            state (int): the index of the current state
            action (int): the integer representation of the action
            reward (int): the reward for the action
            next_state (int): the index of the next state
        """
        if state == DEBUG_STATE and action == DEBUG_ACTION:
            logger.error("Debug action")
        next_state = self.state_action_to_next_state_map[state, action]
        self.q_table[state, action] = self.q_table[
            state, action
        ] + self.learning_rate * (
            self.rewards[next_state]
            + self.gamma * self.max_expected_value(next_state)
            - self.q_table[state, action]
        )

    def max_expected_value(self, state: int) -> np.float16:
        """Returns the max expected value

        Args:
            state (int): the index of the current state

        Returns:
            np.float16: the max expected value
        """
        available_actions = self.available_actions_cache[state]
        which_players_turn = self.which_players_turn(self.index_to_state_map[state])
        if len(available_actions) == 0:
            return np.float16(0)
        if which_players_turn == -1:
            return np.min(self.q_table[state, available_actions])
        return np.max(self.q_table[state, available_actions])

    def q_learning(self, num_episodes=1 * 10**6) -> None:
        """Runs the Q-learning algorithm"""
        logger.debug(f"Starting Q-learning {self.q_table[DEBUG_STATE, DEBUG_ACTION]}")
        for _ in track(range(num_episodes), description="Q-learning", show_speed=True):
            state_index: int = rng.integers(0, self.num_states, dtype=int)
            if state_index == DEBUG_STATE:
                logger.debug(
                    f"Debug state and action Q value {self.q_table[DEBUG_STATE, DEBUG_ACTION]}",
                )
            state = self.index_to_state_map[state_index]
            while self.valid_states[state_index]:
                if self.is_terminal_state(state):
                    break
                random_action: int = rng.choice(
                    self.available_actions_cache[state_index]
                )
                self.update_q_table(state_index, random_action)
                state_index = self.state_action_to_next_state_map[
                    state_index, random_action
                ]
                state = self.index_to_state_map[state_index]

        logger.info("Finished Q-learning")
