# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import math
import typing
from mctsV2 import GameState, mcts_search


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion":
        "1",
        "author":
        "uber_snake_master_360_noscopes",
        "color":
        "#39FF14",  # Color: neon green
        "head":
        "silly",
        "tail":
        "dragon",
        "version":
        "2.3.15.X-alpha_beta_omega_ultimate_edition_360_no_scopes.FINAL-VERSION"
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("Start Game:")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("End Game.\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    """This function retrieves game_state, a dictionary containing all game data, 3 dicts and an int. Here's what it contains:
    * game_state["turn"] returns an int, the current turn number of the game (like the current tick number)
    * game_state["you"] returns the dictionary of data for our snake (a Battlesnake object), like coordinates of each piece              of the body, its length etc... 
            see the details in the docs: https://docs.battlesnake.com/api/objects/battlesnake
    * game_state["board"] returns the dictionary of data regarding the game board (a Board object), like the height and width of         the board, the coordinates of food, and the Battlesnake-objects of all snakes in the game etc... 
            see the details in the docs: https://docs.battlesnake.com/api/objects/board
    * game_state["game"] returns the dictionary of data regarding the game (a Game object), like the id of the game, the ruleset         of the game (also a dict, it contains a dict "settings" about more specific set rules for this game), and the timeout            time for the game (accessed as game_state["game"][timeout]) etc... 
            see the details in the docs - Game object: https://docs.battlesnake.com/api/objects/game
                                        - Ruleset object: https://docs.battlesnake.com/api/objects/ruleset
                                        - Settings object: https://docs.battlesnake.com/api/objects/ruleset-settings
    Example of Move Request (the game_state received by move()) and Move Response (move()'s return value, new move dict):                 https://docs.battlesnake.com/api/example-move    """

    board = game_state["board"]
    you_id = game_state["you"]["id"]
    state = GameState(board, you_id)

    next_move = mcts_search(state, time_limit=0.4)
    
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move, "shout": "I'm a snake! Hissss!"}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
'''
PROJECT NOTES - BATTLE SNAKER:
* Read API-docs: https://docs.battlesnake.com/api

* Use Minimax (swap for Monte Carlo Tree Search?) w. alpha, beta pruning to determine best move out of possible future states
    We have to check how much time we have between server pings to calculate stuff. When 4 snakes remain, maybe just check       for states 2 moves ahead, but when less snakes remain, check for states 3 moves ahead...?

* (TODO Josef) Implement enemy AI used in Monte Carlo tree search (based on Rami's new classes/ internal game state): basic A* for pathfinding to food items + use flood fill to determine if likely moving into a dead end that cannot be escaped. Add anything else?

* Two roles for our two snakes:
   We'll run basically the same script on both snakes, but with different heuristics in the minimax function for each snake.
   Offensive snake - offensive strategy, tries to eat a lot of food, and tries to kill other snakes
   Defensive snake - defensive strategy, tries to stay alive and not get eaten by other snakes

'''
