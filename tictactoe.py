# import modules
import pygame
import numpy as np
from functools import reduce
from typing import List, Tuple

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


class TicTacToe:
    def __init__(self, board_size: int = 3, win_condition: int = 3) -> None:
        self.clicked = False
        self.player = 1
        self.markers = np.zeros((board_size, board_size), dtype=np.int8)
        self.game_over = False
        self.winner = 0

        self.board_size = board_size
        self.win_condition = win_condition

        self.num_cells = self.board_size**2

        try:
            self._initialize_transition_matrix()
        except MemoryError:
            print("Memory Error: Transition Matrix is too large")
            print("Using random function instead")
            self.random_function = True

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

    def check_win(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                col_sum = sum(self.markers[i, j : j + self.win_condition])
                row_sum = sum(self.markers[i : i + self.win_condition, j])
                diag_sum = 0
                anti_diag_sum = 0
                for k in range(self.win_condition):
                    if i + k < self.board_size and j + k < self.board_size:
                        diag_sum += self.markers[i + k, j + k]
                    if i + k < self.board_size and j - k >= 0:
                        anti_diag_sum += self.markers[i + k, j - k]

                if (
                    col_sum == self.win_condition
                    or row_sum == self.win_condition
                    or diag_sum == self.win_condition
                    or anti_diag_sum == self.win_condition
                ):
                    self.game_over = True
                    self.winner = 1
                elif (
                    col_sum == -self.win_condition
                    or row_sum == -self.win_condition
                    or diag_sum == -self.win_condition
                    or anti_diag_sum == -self.win_condition
                ):
                    self.game_over = True
                    self.winner = 2

        # check for tie
        abs_markers = np.absolute(self.markers)
        if np.sum(abs_markers) == self.board_size**2:
            print(self.markers)
            self.game_over = True
            self.winner = 0

    def draw_game_over(self, winner):
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

    def place_marker(self, x: int, y: int, player: int) -> None:
        if self._is_legal_action(self.markers, x, y):
            self.markers[x][y] = player
            self.player = -self.player
            self.check_win()

    def _is_legal_action(self, state: np.ndarray, x: int, y: int) -> bool:
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

    def _index_to_state(self, index: int) -> np.ndarray:
        """A utility function to convert index to state
        Args:
            index (int): index of the state from the state space

        Returns:
            np.ndarray: the current state of the board
        """
        state = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        for i in range(self.board_size):
            for j in range(self.board_size):
                state[i][j] = index % 3
                index = index // 3
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

    def _available_actions(self, state: np.ndarray) -> np.ndarray:
        """A utility function to get the available actions
        Args:
            state (np.ndarray): the current state of the board

        Returns:
            np.ndarray: the available actions
        """

        return np.argwhere(state == 0)

    def _initialize_transition_matrix(self):
        """A utility function to initialize the transition matrix"""
        self.transition_matrix = np.zeros(
            (3**self.num_cells, 3**self.num_cells), dtype=np.float16
        )
        for i in range(3**self.num_cells):
            marker = 1 if i % 2 == 0 else -1
            state = self._index_to_state(i)
            available_actions = self._available_actions(state)
            for action in available_actions:
                next_state = state.copy()
                next_state[action[0]][action[1]] = marker
                next_state_index = self.state_to_index(next_state)
                self.transition_matrix[i][next_state_index] = 1 / len(available_actions)

            # normalize the row
            self.transition_matrix[i] /= self.transition_matrix[i].sum()
            val = np.sum(self.transition_matrix[i])
            # check the first non-zero index in the row
            first_non_zero_index = np.argwhere(self.transition_matrix[i] != 0)[0][0]
            self.transition_matrix[i][first_non_zero_index] = (
                1
                - np.sum(self.transition_matrix[i])
                + self.transition_matrix[i][first_non_zero_index]
            )

            if np.isnan(self.transition_matrix[i].sum()):
                self.transition_matrix[i] = np.zeros(3**self.num_cells)
            elif np.sum(self.transition_matrix[i]) != 1:
                print(
                    "Error in transition matrix",
                    i,
                    self.transition_matrix[i].sum(),
                    val,
                )

    def reset(self):
        self.markers = np.zeros((self.board_size, self.board_size))
        self.player = 1
        self.game_over = False
        self.winner = 0
        try:
            self._initialize_transition_matrix()
        except:
            # silent ignore the exception because it was already raised in the constructor
            pass
        self.clicked = False

    def transition_function(self, action, next_state) -> float:
        current_state_index = self.state_to_index(self.markers)
        next_state_index = self.state_to_index(next_state)
        available_actions = np.argwhere(self.markers == 0)
        if len(available_actions) == 0:
            return 0
        if action not in available_actions:
            return 0

        return self.transition_matrix[current_state_index][next_state_index]


def main():
    # initialize pygame
    game = TicTacToe(4, 3)

    # set up clock (for efficiency)
    clock = pygame.time.Clock()
    # main loop
    run = True
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

            if game.random_function is False:
                probabilities = game.transition_matrix[
                    game.state_to_index(game.markers)
                ]
                if probabilities.sum() == 0:
                    # game is over
                    game.check_win()
                else:
                    next_state_index = int(rng.multinomial(1, probabilities).argmax())

                    next_state = game._index_to_state(next_state_index)
                    opponent_action = game.get_action(game.markers, next_state)

                    x, y = opponent_action
                    game.place_marker(x, y, game.player)
                    game.check_win()
            else:
                random_action = rng.choice(game._available_actions(game.markers))
                x, y = random_action
                game.place_marker(x, y, game.player)
                game.check_win()

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
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
