import numpy as np
import pygame

from tictactoe import SCALING_FACTOR, FPS
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration


def main(
    game: ValueIteration | PolicyIteration,
    load_policy: bool = False,
):
    if load_policy:
        with open("policy.npy", "rb") as f:
            policy = np.load(f)
    else:
        policy, _ = game.get_policy()
        with open("policy.npy", "wb") as f:
            np.save(f, policy)

    # # set up clock (for efficiency)
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
                # get next move
                state = game.state_to_index(game.markers)
                action = policy[state]
                if not game.game_over and action[0] == -1 and action[1] == -1:
                    print("No action found for state", game.markers)
                game.place_marker(action[0], action[1])
                game_over, winner = game.check_win(game.markers, game.win_condition)
                game.game_over = game_over
                game.winner = winner

        # check if game has been won
        if game.game_over == True:
            game.draw_markers()
            game.draw_game_over(game.winner)
            pygame.display.update()

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
        # game.draw_markers()
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    game = PolicyIteration(3, 3)
    main(game)
