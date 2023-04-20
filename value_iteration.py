import pickle
import pygame
import numpy as np
import numpy.typing as npt
from tictactoe import TicTacToe, FPS, SCALING_FACTOR
from typing import Tuple
from copy import deepcopy

DEBUG_STATE = -1


class ValueIteration(TicTacToe):
    def __init__(
        self, board_size: int = 3, win_condition: int = 3, gamma=np.float16(0.9)
    ) -> None:
        super().__init__(board_size, win_condition, gamma)
        self.gamma = gamma

    @staticmethod
    def expected_value(
        probabilities: npt.NDArray[np.float16], value: npt.NDArray[np.float16]
    ) -> np.float16:
        """A utility function to calculate the expected value

        Args:
            probabilities (np.ndarray): the probabilities of the actions
            rewards (np.ndarray): the rewards of the actions

        Returns:
            float: the expected value
        """
        return np.sum(probabilities * value)

    def update_values(
        self, epsilon: float = 1e-4, max_iterations: int = 100
    ) -> npt.NDArray[np.float16]:
        """Performs value update

        Args:
            epsilon (float, optional): the limiting difference. Defaults to 1e-4.
            max_iterations (int, optional): max number of iterations the algorithm is allowed to run. Defaults to 100.

        Returns:
            npt.NDArray[np.float16]: an array of values for each state
        """
        values = np.zeros(3**self.num_cells).astype(np.float16)

        delta = np.inf
        num_iterations = 0
        while num_iterations < max_iterations:
            print("iteration", num_iterations, "delta", delta)
            values_copy = values.copy()
            for i in range(3**self.num_cells):
                player = self.which_players_turn(self.index_to_state(i))
                if not self.is_valid_state(self.index_to_state(i)):
                    continue
                if i == DEBUG_STATE:
                    print("debug state", player)
                current_state = self.index_to_state(i)
                max_expected_value = -np.inf if player == 1 else np.inf
                available_actions = self.available_actions(current_state)
                reward = self.reward_function(current_state)
                computed_values = []
                for action in available_actions:
                    next_state = current_state.copy()
                    next_state[action[0]][action[1]] = player
                    next_state_index = self.state_to_index(next_state)
                    value = reward + self.gamma * values_copy[next_state_index]
                    if player == 1:
                        max_expected_value = max(max_expected_value, value)
                    elif player == -1:
                        max_expected_value = min(max_expected_value, value)
                    computed_values.append(value)

                player = -player

                if i == DEBUG_STATE:
                    print("available actions", available_actions.tolist())
                    print("computed values", computed_values)

                if len(available_actions) == 0 and self.is_valid_state(current_state):
                    # if the state is terminal, then the expected value is 0
                    max_expected_value = self.reward_function(current_state)
                elif len(available_actions) == 0:
                    max_expected_value = 0

                values[i] = max_expected_value
            delta = min(delta, np.max(np.abs(values_copy - values)))

            num_iterations += 1

            if delta < epsilon:
                break

        print("Value iteration converged after", num_iterations, "iterations to", delta)

        return values

    def get_policy(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates a policy based on value iteration

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: A tuple of policy and values
        """
        policy = np.array([(-1, -1)] * 3**self.num_cells)
        values = self.update_values()
        for i in range(3**self.num_cells):
            player = self.which_players_turn(self.index_to_state(i))
            if i == DEBUG_STATE:
                print("debug state")
            best_action = None
            max_expected_value = -np.inf if player == 1 else np.inf
            current_state = self.index_to_state(i)
            available_actions = self.available_actions(current_state)
            for action in available_actions:
                next_state = self.index_to_state(i).copy()
                next_state[action[0]][action[1]] = player
                next_state_index = self.state_to_index(next_state)
                if player == 1 and values[next_state_index] > max_expected_value:
                    best_action = action
                    max_expected_value = values[next_state_index]
                elif player == -1 and values[next_state_index] < max_expected_value:
                    best_action = action
                    max_expected_value = values[next_state_index]

            if best_action is None:
                # print(f"Could not find a best action for the state {current_state}")
                pass
            else:
                policy[i] = best_action

            player = -player

        return policy, values
