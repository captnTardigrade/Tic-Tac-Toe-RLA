import os
import logging

import numpy as np
import pandas as pd
import pygame

from tictactoe import SCALING_FACTOR, FPS, rng
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from q_learning import QLearning

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s", datefmt='%m/%d/%Y %H:%M:%S',)

file_handler = logging.FileHandler("logs/game.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def main(
    game: ValueIteration | PolicyIteration,
    load_policy: bool = False,
):
    if load_policy:
        with open("policy.npy", "rb") as f:
            policy = np.load(f)
        with open("values.npy", "rb") as f:
            values = np.load(f)
    else:
        policy, values = game.get_policy()
        with open("policy.npy", "wb") as f:
            np.save(f, policy)
        with open("values.npy", "wb") as f:
            np.save(f, values)

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
                    logger.warn("No action found for state", game.markers)
                if not game.game_over and not game.is_legal_action(game.markers, action[0], action[1]):
                    logger.error(f"State {state}")
                    logger.error(f"Q Values {values[state]}")
                    logger.error(f"Illegal action {action}")
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


def game_against_random_agent(
    game: ValueIteration | PolicyIteration,
    num_games: int = 100,
    load_policy: bool = False,
) -> pd.DataFrame:
    """Runs num_games number of games against a random agent

    Args:
        game (ValueIteration | PolicyIteration): the type of algorithm to use to compute the policy
        num_games (int, optional): number of games to run. Defaults to 100.
        load_policy (bool, optional): whether to load the policy from cache or to find one. Defaults to False.

    Returns:
        pd.DataFrame: a dataframe containing the game number, winner, and the final board state
    """
    game_stats = []
    if load_policy:
        with open("policy.npy", "rb") as f:
            policy = np.load(f)
        with open("values.npy", "rb") as f:
            values = np.load(f)
    else:
        policy, values = game.get_policy()
        with open("policy.npy", "wb") as f:
            np.save(f, policy)
        with open("values.npy", "wb") as f:
            np.save(f, values)
    run = True
    current_game_number = 0
    while run and current_game_number < num_games:
        # draw board and markers first
        game.draw_board()
        game.draw_markers()

        # handle events
        while not game.game_over:
            # run new game
            if game.player == 1:
                available_actions = game.available_actions(game.markers)
                random_action_idx = rng.integers(len(available_actions))
                cell_x, cell_y = available_actions[random_action_idx]
                if game.markers[cell_x][cell_y] == 0:
                    game.place_marker(cell_x, cell_y)
                    game_over, winner = game.check_win(game.markers, game.win_condition)
                    game.game_over = game_over
                    game.winner = winner

            if game.player == -1:
                # get next move
                state = game.state_to_index(game.markers)
                action = policy[state]
                if not game.game_over and action[0] == -1 and action[1] == -1:
                    logger.warn("No action found for state", game.markers)
                if not game.game_over and not game.is_legal_action(game.markers, action[0], action[1]):
                    logger.error(f"Illegal action {action}")
                game.place_marker(action[0], action[1])
                game_over, winner = game.check_win(game.markers, game.win_condition)
                game.game_over = game_over
                game.winner = winner

        # check if game has been won
        if game.game_over == True:
            game_stats.append([current_game_number, game.winner, game.markers])
            current_game_number += 1
            game.draw_markers()
            game.draw_game_over(game.winner)
            pygame.display.update()
            game.reset()

        # update display
        pygame.display.update()

    game_df = pd.DataFrame(game_stats, columns=["game", "winner", "final_state"])

    return game_df


if __name__ == "__main__":
    algorithms = {
        "Value Iteration": ValueIteration(3, 3),
        "Policy Iteration": PolicyIteration(3, 3),
        "Q Learning": QLearning(3, 3),
    }

    if not os.path.exists("stats"):
        os.mkdir("stats")        

    for algorithm, game in algorithms.items():
        logger.info(algorithm)

        results = game_against_random_agent(game, 1000, load_policy=False)

        win_percent_by_player = results.groupby("winner").count()
        win_percent_by_player["percent"] = (
            100 * win_percent_by_player["game"] / win_percent_by_player["game"].sum()
        )

        logger.info(win_percent_by_player)
        win_percent_by_player.to_csv(f"stats/{algorithm}.csv")
        logger.info("*" * 50)