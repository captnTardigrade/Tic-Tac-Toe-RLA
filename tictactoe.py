# import modules
import pygame
import numpy as np

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
font = pygame.font.SysFont(None, 40)


class TicTacToe:
    def __init__(self, board_size: int = 3, win_condition: int = 3) -> None:
        self.clicked = False
        self.player = 1
        self.pos = (0, 0)
        self.markers = []
        self.game_over = False
        self.winner = 0

        self.board_size = board_size
        self.win_condition = win_condition

        self.num_cells = self.board_size ** 2

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

        # create empty BOARD_SIZE x BOARD_SIZE list to represent the grid
        for x in range(self.board_size):
            row = [0] * self.board_size
            self.markers.append(row)

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
        markers = np.array(self.markers)
        for i in range(self.board_size):
            for j in range(self.board_size):
                col_sum = sum(markers[i, j : j + self.win_condition])
                row_sum = sum(markers[i : i + self.win_condition, j])
                diag_sum = 0
                anti_diag_sum = 0
                for k in range(self.win_condition):
                    if i + k < self.board_size and j + k < self.board_size:
                        diag_sum += markers[i + k, j + k]
                    if i + k < self.board_size and j - k >= 0:
                        anti_diag_sum += markers[i + k, j - k]

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
        abs_markers = np.absolute(markers)
        if np.sum(abs_markers) == self.board_size**2:
            self.game_over = True
            self.winner = 0

    def draw_game_over(self, winner):
        TEXT_COLOR = (2, 48, 71)
        if winner != 0:
            end_text = "Player " + str(winner) + " wins!"
        elif winner == 0:
            end_text = "You have tied!"

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

    def place_marker(self, x, y, player) -> bool:
        if self.markers[x][y] == 0:
            self.markers[x][y] = player
            self.player = -self.player
            self.check_win()
            return True
        return False

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
            x, y = rng.choice(game.board_size, 2, replace=True)
            while not game.game_over and not game.place_marker(
                x, y,
                game.player,
            ):
                x, y = rng.choice(game.board_size, 2, replace=True)


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
                        game.game_over = False
                        game.player = 1
                        game.pos = (0, 0)
                        game.markers = []
                        game.winner = 0
                        # create empty BOARD_SIZE x BOARD_SIZE list to represent the grid
                        for _ in range(game.board_size):
                            row = [0] * game.board_size
                            game.markers.append(row)
                mouse_clicked = True

        # update display
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
