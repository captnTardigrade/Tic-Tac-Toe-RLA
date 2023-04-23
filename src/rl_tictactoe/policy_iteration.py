# import modules
import os
import logging
import numpy as np
from collections import defaultdict

# import typing
from typing import Tuple, Dict, List, DefaultDict
import numpy.typing as npt

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s", datefmt='%m/%d/%Y %H:%M:%S',)

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

file_handler = logging.FileHandler(f"{logs_dir}/game.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

from rl_tictactoe import tictactoe

# DEBUGGING
DEBUG_STATE = -1


class PolicyIteration(tictactoe.TicTacToe):
    """Policy Iteration algorithm for Tic-Tac-Toe"""

    def __init__(
        self,
        board_size: int = 3,
        win_condition: int = 3,
        gamma: np.float16 = np.float16(0.9),
    ) -> None:
        super().__init__(board_size, win_condition, gamma)

        self.index_to_state_map: Dict[int, npt.NDArray[np.int8]] = {}
        self.state_to_index_map: Dict[tuple, int] = {}
        self.valid_states = np.zeros(self.num_states, dtype=bool)
        self.rewards = np.zeros(self.num_states, dtype=np.float16)
        self.next_states: DefaultDict[int, List[int]] = defaultdict(list)

        # (state, next_state) -> action
        self.state_next_state_to_action = (
            np.ones((self.num_states, self.num_states), dtype=tuple) * -1
        )

        # (state, action) -> next_state
        self.state_action_to_next_state_map = (
            np.ones((self.num_states, self.num_cells), dtype=int) * -1
        )

        self.available_actions_cache: Dict[int, npt.NDArray[np.int8]] = {}

        for i in np.arange(self.num_states, dtype=int):
            self.index_to_state_map[i] = self.index_to_state(i)
            state = self.index_to_state_map[i]
            self.state_to_index_map[tuple(state.flatten())] = i
            self.valid_states[i] = self.is_valid_state(state)
            if self.valid_states[i]:
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
                    self.next_states[i].append(next_state)

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

        for i in np.arange(self.num_states, dtype=int):
            state = self.index_to_state_map[i]
            if self.valid_states[i]:
                self.available_actions_cache[i] = self.available_actions(state)

        if self.board_size == 3 and self.win_condition == 3:
            assert self.valid_states.sum() == 5478

    @staticmethod
    def action_to_int(action: npt.NDArray[np.int8]) -> int:
        """converts an action to an integer

        Args:
            action (npt.NDArray[np.int8]): the action

        Returns:
            int: the integer representation of the action
        """
        return 3 * action[0] + action[1]

    def update_values(
        self,
        policy: npt.NDArray[np.int8],
        prev_values: npt.NDArray[np.float16],
    ) -> npt.NDArray[np.float16]:
        """Updates the values of the states based on the policy

        Args:
            policy (npt.NDArray[np.int8]): The policy to use
            prev_values (npt.NDArray[np.float16]): The current values for each state

        Returns:
            npt.NDArray[np.float16]: The updated values for each state
        """
        actions = 3 * policy[:, 0] + policy[:, 1]
        next_states = self.state_action_to_next_state_map[
            np.arange(self.num_states), actions
        ]

        next_states_values = prev_values[next_states]
        values = self.rewards + self.gamma * next_states_values

        return values

    def get_policy(
        self, epsilon: float = 1e-4, max_iterations: int = 100
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates a policy based on value iteration

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: A tuple of policy and values
        """
        policy = np.zeros((self.num_states, 2), dtype=np.int8)
        values = self.update_values(
            policy, np.zeros(self.num_states).astype(np.float16)
        )

        delta = np.inf
        num_iterations = 0
        while num_iterations < max_iterations:
            policy_copy = policy.copy()
            prev_values = values.copy()
            values = self.update_values(policy, prev_values)
            for i in range(self.num_states):
                state = self.index_to_state_map[i]
                if not self.valid_states[i]:
                    continue

                actions = self.available_actions_cache[i]
                if len(actions) == 0:
                    continue

                action_values = np.zeros(len(actions))
                action_ints = 3 * actions[:, 0] + actions[:, 1]

                next_state_indices = self.state_action_to_next_state_map[i, action_ints]
                action_values = values[next_state_indices]

                if self.which_players_turn(state) == 1:
                    policy[i] = actions[np.argmax(action_values)]
                else:
                    policy[i] = actions[np.argmin(action_values)]

            delta = np.max(np.abs(policy_copy - policy))
            if delta < epsilon:
                break

            logger.info(f"iteration: {num_iterations} delta: {delta}" )
            num_iterations += 1

        logger.info(
            f"Policy iteration convereged to delta: {delta} in {num_iterations} iterations"
        )

        return policy, values
