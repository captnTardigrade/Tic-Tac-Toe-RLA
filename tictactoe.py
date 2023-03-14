# import modules
import os
import pygame
import numpy as np
from functools import reduce

# import typing
from typing import Tuple
import numpy.typing as npt

SEED = 1337

rng = np.random.default_rng(SEED)

pygame.init()

FPS = 60

LINE_WIDTH = 6

SCALING_FACTOR = 100

# define colours
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

DIALOG_COLOR = (205, 180, 219)
X_MARKER_COLOR = (247, 37, 133)
O_MARKER_COLOR = (76, 201, 240)

# define font
font = pygame.font.SysFont("Arial", 40)

# DEBUGGING
DEBUG_STATE = 10258


class TicTacToe:
    """A Tic-Tac-Toe game"""

    def __init__(
        self, board_size: int = 3, win_condition: int = 3, gamma: float = 0.9
    ) -> None:
        self.clicked = False
        self.player = 1
        self.markers = np.zeros((board_size, board_size), dtype=np.int8)
        self.game_over = False
        self.winner = 0
        self.gamma = gamma

        self.board_size = board_size
        self.win_condition = win_condition

        self.num_cells = self.board_size**2

        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = (
            self.board_size * SCALING_FACTOR,
            self.board_size * SCALING_FACTOR,
        )

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe")

        # setup a rectangle for "Play Again" Option
        self.again_rect = pygame.Rect(
            self.SCREEN_WIDTH // 2 - 80, self.SCREEN_HEIGHT // 2, 160, 50
        )

    def draw_board(self):
        """draw the board on the screen"""
        bg = (255, 218, 233)
        grid = (42, 52, 57)
        self.screen.fill(bg)
        for x in range(1, self.board_size):
            pygame.draw.line(
                self.screen,
                grid,
                (0, SCALING_FACTOR * x),
                (self.SCREEN_WIDTH, SCALING_FACTOR * x),
                LINE_WIDTH,
            )
            pygame.draw.line(
                self.screen,
                grid,
                (SCALING_FACTOR * x, 0),
                (SCALING_FACTOR * x, self.SCREEN_HEIGHT),
                LINE_WIDTH,
            )

    def draw_markers(self):
        """draw markers on the screen"""
        x_pos = 0
        for x in self.markers:
            y_pos = 0
            for y in x:
                if y == 1:
                    pygame.draw.line(
                        self.screen,
                        X_MARKER_COLOR,
                        (x_pos * 100 + 15, y_pos * 100 + 15),
                        (x_pos * 100 + 85, y_pos * 100 + 85),
                        LINE_WIDTH,
                    )
                    pygame.draw.line(
                        self.screen,
                        X_MARKER_COLOR,
                        (x_pos * 100 + 85, y_pos * 100 + 15),
                        (x_pos * 100 + 15, y_pos * 100 + 85),
                        LINE_WIDTH,
                    )
                if y == -1:
                    pygame.draw.circle(
                        self.screen,
                        O_MARKER_COLOR,
                        (x_pos * 100 + 50, y_pos * 100 + 50),
                        38,
                        LINE_WIDTH,
                    )
                y_pos += 1
            x_pos += 1

    def draw_game_over(self, winner: int):
        """draws the game over screen"""
        TEXT_COLOR = (2, 48, 71)
        end_text = "You have tied!"
        if winner != 0:
            end_text = "Player " + str(winner) + " wins!"

        end_img = font.render(end_text, True, TEXT_COLOR)
        pygame.draw.rect(
            self.screen,
            DIALOG_COLOR,
            (self.SCREEN_WIDTH // 2 - 100, self.SCREEN_HEIGHT // 2 - 60, 200, 50),
        )
        self.screen.blit(
            end_img, (self.SCREEN_WIDTH // 2 - 100, self.SCREEN_HEIGHT // 2 - 50)
        )

        again_text = "Play Again?"
        again_img = font.render(again_text, True, TEXT_COLOR)
        pygame.draw.rect(self.screen, DIALOG_COLOR, self.again_rect)
        self.screen.blit(
            again_img, (self.SCREEN_WIDTH // 2 - 80, self.SCREEN_HEIGHT // 2 + 10)
        )

    @staticmethod
    def check_win(state: npt.NDArray[np.int8], win_condition: int) -> Tuple[bool, int]:
        """checks if there is a winner

        Args:
            state (npt.NDArray[np.int8]): the board
            win_condition (int): the number of markers in a row to win

        Returns:
            Tuple[bool, int]: (game_over, winner)
        """
        board_size = state.shape[0]
        game_over = False
        winner = 0

        row_sum = state.sum(axis=1)
        col_sum = state.sum(axis=0)

        if win_condition in row_sum:
            game_over = True
            winner = 1
            return game_over, winner

        if -win_condition in row_sum:
            game_over = True
            winner = 2
            return game_over, winner

        if win_condition in col_sum:
            game_over = True
            winner = 1
            return game_over, winner

        if -win_condition in col_sum:
            game_over = True
            winner = 2
            return game_over, winner

        for i in range(board_size):
            for j in range(board_size):
                diag_sum = 0
                anti_diag_sum = 0
                for k in range(win_condition):
                    if i + k < board_size and j + k < board_size:
                        diag_sum += state[i + k, j + k]
                    if i + k < board_size and j - k >= 0:
                        anti_diag_sum += state[i + k, j - k]

                if diag_sum == win_condition or anti_diag_sum == win_condition:
                    game_over = True
                    winner = 1
                elif diag_sum == -win_condition or anti_diag_sum == -win_condition:
                    game_over = True
                    winner = 2

        # check for tie
        abs_markers = np.absolute(state)
        if np.sum(abs_markers) == board_size**2:
            game_over = True
            winner = 0

        return game_over, winner

    def place_marker(self, x: int, y: int) -> None:
        """Places a marker on the board and updates the board inplace. Also updates the current player
        Args:
            x (int): column
            y (int): row
            player (int): current player [1 -> player 1 (X), -1 -> player 2(O)]
        """
        if self.is_legal_action(self.markers, x, y):
            self.markers[x][y] = self.player
            self.player = -self.player

    @staticmethod
    def is_legal_action(state: np.ndarray, x: int, y: int) -> bool:
        """A utility function to check if the action is legal"""
        return state[x][y] == 0

    @staticmethod
    def state_to_index(state: np.ndarray) -> int:
        """A utility function to convert state to index
        Args:
            state (np.ndarray): the current state of the board

        Returns:
            int: index of the state from the state space
        """
        state[state == -1] = 2
        val = int(reduce(lambda acc, curr: acc * 3 + curr, state.flatten()))
        state[state == 2] = -1
        return val

    def index_to_state(self, index: int) -> np.ndarray:
        """A utility function to convert index to state
        Args:
            index (int): index of the state from the state space

        Returns:
            np.ndarray: the current state of the board
        """
        state = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        base_3_repr = np.base_repr(index, base=3).zfill(self.board_size**2)
        state = np.array([int(i) for i in base_3_repr]).reshape(
            (self.board_size, self.board_size)
        )
        state[state == 2] = -1
        return state

    @staticmethod
    def get_action(state: np.ndarray, next_state: np.ndarray) -> Tuple[int, int]:
        """A utility function to get the action from the state and next state
        Args:
            state (np.ndarray): the current state of the board
            next_state (np.ndarray): the next state of the board

        Returns:
            Tuple[int, int]: the action
        """
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i][j] != next_state[i][j]:
                    return i, j

        return (-1, -1)

    def available_actions(self, state: np.ndarray) -> np.ndarray:
        """A utility function to get the available actions
        Args:
            state (np.ndarray): the current state of the board

        Returns:
            np.ndarray: the available actions
        """
        game_over, _ = self.check_win(state, self.win_condition)
        if game_over:
            return np.array([])
        return np.argwhere(state == 0)

    def reset(self):
        """resets the game"""
        self.markers = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.player = 1
        self.game_over = False
        self.winner = 0
        self.clicked = False

    def is_valid_state(self, state: npt.NDArray[np.int8]) -> bool:
        """A utility function to check if the terminal state is possible
        Args:
            state (npt.NDArray[np.int8]): the state to be checked

        Returns:
            bool: Returns true if the state is possible
        """
        num_ones = np.sum(state == 1)
        num_minus_ones = np.sum(state == -1)
        diff = num_ones - num_minus_ones
        if 0 > diff or diff > 1:
            return False

        rows_sum = np.sum(state, axis=1)
        if self.win_condition in rows_sum and -self.win_condition in rows_sum:
            return False

        cols_sum = np.sum(state, axis=0)
        if self.win_condition in cols_sum and -self.win_condition in cols_sum:
            return False

        for i in range(self.board_size):
            for j in range(self.board_size):
                diag_sum = 0
                anti_diag_sum = 0
                for k in range(self.win_condition):
                    if i + k < self.board_size and j + k < self.board_size:
                        diag_sum += self.markers[i + k, j + k]
                    if i + k < self.board_size and j - k >= 0:
                        anti_diag_sum += self.markers[i + k, j - k]

                if (
                    diag_sum == self.win_condition
                    and anti_diag_sum == -self.win_condition
                ):
                    return False

        return True

    def reward_function(self, state: npt.NDArray[np.int8]) -> float:
        """A utility function to get the reward of the a state

        Returns:
            float: the reward of the current state
        """
        game_over, winner = TicTacToe.check_win(state, self.win_condition)
        if game_over:
            if winner == 0:
                return 0
            elif winner == 1:
                return 10
            else:
                return -10
        return 0

    def which_players_turn(self, state: npt.NDArray[np.int8]) -> int:
        """A utility function to get the current player

        Returns:
            int: the current player
        """
        num_ones = np.sum(state == 1)
        num_minus_ones = np.sum(state == -1)
        diff = num_ones - num_minus_ones
        if diff == 0:
            return 1
        else:
            return -1

    def update_values(
        self,
        state: npt.NDArray[np.int8],
        policy: npt.NDArray[np.int8],
        prev_values: npt.NDArray[np.float16],
    ) -> npt.NDArray[np.float16]:
        values = np.zeros(3**self.num_cells).astype(np.float16)
        for i in range(3**self.num_cells):
            reward = self.reward_function(self.index_to_state(i))
            next_state = self.index_to_state(i)
            action_x, action_y = policy[i]
            next_state[action_x][action_y] = self.which_players_turn(state)
            next_state_index = self.state_to_index(next_state)
            if not self.is_valid_state(next_state):
                values[i] = 0
            else:
                values[i] = reward + self.gamma * prev_values[next_state_index]

        return values

    def get_policy(
        self, epsilon: float = 1e-4, max_iterations: int = 100
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates a policy based on value iteration

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: A tuple of policy and values
        """
        policy = rng.integers(0, self.board_size, size=(3**self.num_cells, 2), dtype=np.int8)
        values = self.update_values(
            self.markers, policy, np.zeros(3**self.num_cells).astype(np.float16)
        )

        delta = np.inf
        num_iterations = 0
        while num_iterations < max_iterations:
            for i in range(3**self.num_cells):
                policy_copy = policy.copy()
                prev_values = values.copy()
                values = self.update_values(self.markers, policy, prev_values)

                state = self.index_to_state(i)
                actions = self.available_actions(state)
                if len(actions) == 0:
                    continue
                if not self.is_valid_state(state):
                    continue
                action_values = np.zeros(len(actions))
                for j, action in enumerate(actions):
                    next_state = state.copy()
                    next_state[action[0]][action[1]] = self.which_players_turn(state)
                    next_state_index = self.state_to_index(next_state)
                    action_values[j] = values[next_state_index]

                if self.which_players_turn(state) == 1:
                    policy[i] = actions[np.argmax(action_values)]
                else:
                    policy[i] = actions[np.argmin(action_values)]

                delta = np.max(policy_copy - policy)
                if delta < epsilon:
                    break

            print("iteration:", num_iterations, "delta:", delta)
            num_iterations += 1
            
        print(f"Policy iteration convereged to delta: {delta} in {num_iterations} iterations")

        return policy, values


def main(load_policy: bool = False):
    # initialize pygame
    game = TicTacToe(3, 3)
    if load_policy:
        with open("policy.npy", "rb") as f:
            policy = np.load(f)
    else:
        policy, values = game.get_policy()
        with open("policy.npy", "wb") as f:
            np.save(f, policy)

    # set up clock (for efficiency)
    clock = pygame.time.Clock()
    # main loop
    run = True
    clicked = False
    while run:
        clock.tick(FPS)
        # draw board and markers first
        game.draw_board()
        game.draw_markers()

        # handle events
        for event in pygame.event.get():
            # handle game exit
            if event.type == pygame.QUIT:
                run = False

            if game.game_over:
                continue

            # run new game
            if game.player == 1:
                # check for mouseclick
                if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
                    clicked = True
                if event.type == pygame.MOUSEBUTTONUP and clicked == True:
                    clicked = False
                    p_x, p_y = pygame.mouse.get_pos()
                    cell_x = p_x // SCALING_FACTOR
                    cell_y = p_y // SCALING_FACTOR
                    # check if the mouse click is not in the board
                    if cell_x >= game.board_size or cell_y >= game.board_size:
                        continue
                    if game.markers[cell_x][cell_y] == 0:
                        game.place_marker(cell_x, cell_y)
                        game_over, winner = game.check_win(
                            game.markers, game.win_condition
                        )
                        game.game_over = game_over
                        game.winner = winner

            if game.player == -1:
                print("Player 2's turn")
                # get next move
                state = game.state_to_index(game.markers)
                print(game.markers)
                action = policy[state]
                if action[0] == -1 and action[1] == -1:
                    print("No action found for state", game.markers)
                game.place_marker(action[0], action[1])
                game_over, winner = game.check_win(game.markers, game.win_condition)
                game.game_over = game_over
                game.winner = winner

        # check if game has been won
        if game.game_over == True:
            game.draw_game_over(game.winner)

            mouse_clicked = False
            while not mouse_clicked:
                event = pygame.event.wait()
                # check for mouseclick to see if we clicked on Play Again
                if event.type == pygame.MOUSEBUTTONDOWN and game.clicked == False:
                    game.clicked = True
                if event.type == pygame.MOUSEBUTTONUP and game.clicked == True:
                    game.clicked = False
                    pos = pygame.mouse.get_pos()
                    if game.again_rect.collidepoint(pos):
                        # reset variables
                        game.reset()
                mouse_clicked = True

        # update display
        game.draw_markers()
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    DEBUG_STATE = -1
    main(load_policy=False)
    # game = TicTacToe(3, 3)

    # state = np.array([
    #     [1, -1, -1],
    #     [-1, -1, 1],
    #     [1, 1, 1]
    # ])

    # print(game.check_win(state, 3))

    # print(game.which_players_turn(state))

    # state_index = game.state_to_index(state)

    # DEBUG_STATE = -1

    # policy, values = game.get_policy()

    # print(policy[state_index])
