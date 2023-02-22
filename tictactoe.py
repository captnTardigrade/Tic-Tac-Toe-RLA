# import modules
import pygame
import numpy as np

pygame.init()

LINE_WIDTH = 6
BOARD_SIZE = 5
WIN_CONDITION = 4

SCALING_FACTOR = 100

SCREEN_WIDTH, SCREEN_HEIGHT = BOARD_SIZE * SCALING_FACTOR, BOARD_SIZE * SCALING_FACTOR

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe")

# define colours
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# define font
font = pygame.font.SysFont(None, 40)

# define variables
clicked = False
player = 1
pos = (0, 0)
markers = []
game_over = False
winner = 0

# setup a rectangle for "Play Again" Option
again_rect = pygame.Rect(SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2, 160, 50)

# create empty BOARD_SIZE x BOARD_SIZE list to represent the grid
for x in range(BOARD_SIZE):
    row = [0] * BOARD_SIZE
    markers.append(row)


def draw_board():
    bg = (255, 255, 210)
    grid = (50, 50, 50)
    screen.fill(bg)
    for x in range(1, BOARD_SIZE):
        pygame.draw.line(
            screen,
            grid,
            (0, SCALING_FACTOR * x),
            (SCREEN_WIDTH, SCALING_FACTOR * x),
            LINE_WIDTH,
        )
        pygame.draw.line(
            screen,
            grid,
            (SCALING_FACTOR * x, 0),
            (SCALING_FACTOR * x, SCREEN_HEIGHT),
            LINE_WIDTH,
        )


def draw_markers():
    x_pos = 0
    for x in markers:
        y_pos = 0
        for y in x:
            if y == 1:
                pygame.draw.line(
                    screen,
                    RED,
                    (x_pos * 100 + 15, y_pos * 100 + 15),
                    (x_pos * 100 + 85, y_pos * 100 + 85),
                    LINE_WIDTH,
                )
                pygame.draw.line(
                    screen,
                    RED,
                    (x_pos * 100 + 85, y_pos * 100 + 15),
                    (x_pos * 100 + 15, y_pos * 100 + 85),
                    LINE_WIDTH,
                )
            if y == -1:
                pygame.draw.circle(
                    screen, GREEN, (x_pos * 100 + 50, y_pos * 100 + 50), 38, LINE_WIDTH
                )
            y_pos += 1
        x_pos += 1


def check_win():
    global markers, game_over, winner
    markers = np.array(markers)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            col_sum = sum(markers[i, j : j + WIN_CONDITION])
            row_sum = sum(markers[i : i + WIN_CONDITION, j])
            diag_sum = 0
            for k in range(WIN_CONDITION):
                if i + k < BOARD_SIZE and j + k < BOARD_SIZE:
                    diag_sum += markers[i + k, j + k]

            if (
                col_sum == WIN_CONDITION
                or row_sum == WIN_CONDITION
                or diag_sum == WIN_CONDITION
            ):
                game_over = True
                winner = 1
            elif (
                col_sum == -WIN_CONDITION
                or row_sum == -WIN_CONDITION
                or diag_sum == -WIN_CONDITION
            ):
                game_over = True
                winner = 2


def draw_game_over(winner):
    if winner != 0:
        end_text = "Player " + str(winner) + " wins!"
    elif winner == 0:
        end_text = "You have tied!"

    end_img = font.render(end_text, True, BLUE)
    pygame.draw.rect(
        screen, GREEN, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 60, 200, 50)
    )
    screen.blit(end_img, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50))

    again_text = "Play Again?"
    again_img = font.render(again_text, True, BLUE)
    pygame.draw.rect(screen, GREEN, again_rect)
    screen.blit(again_img, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 + 10))


def main():
    global clicked, player, pos, markers, game_over, winner
    # main loop
    run = True
    while run:
        # draw board and markers first
        draw_board()
        draw_markers()

        # handle events
        for event in pygame.event.get():
            # handle game exit
            if event.type == pygame.QUIT:
                run = False
            # run new game
            if not game_over:
                # check for mouseclick
                if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
                    clicked = True
                if event.type == pygame.MOUSEBUTTONUP and clicked == True:
                    clicked = False
                    pos = pygame.mouse.get_pos()
                    cell_x = pos[0] // SCALING_FACTOR
                    cell_y = pos[1] // SCALING_FACTOR
                    if markers[cell_x][cell_y] == 0:
                        markers[cell_x][cell_y] = player
                        player *= -1
                        check_win()

        # check if game has been won
        if game_over == True:
            draw_game_over(winner)
            # check for mouseclick to see if we clicked on Play Again
            if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
                clicked = True
            if event.type == pygame.MOUSEBUTTONUP and clicked == True:
                clicked = False
                pos = pygame.mouse.get_pos()
                if again_rect.collidepoint(pos):
                    # reset variables
                    game_over = False
                    player = 1
                    pos = (0, 0)
                    markers = []
                    winner = 0
                    # create empty BOARD_SIZE x BOARD_SIZE list to represent the grid
                    for _ in range(BOARD_SIZE):
                        row = [0] * BOARD_SIZE
                        markers.append(row)

        # update display
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
