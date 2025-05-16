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

import typing
from SnakeAI import SnakeAI

snake_ai = SnakeAI() # Create an instance of the SnakeAI class

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion":
        "1",
        "Josepi The Moustache & Ram The Ham":
        "Johnny SilverSnake",
        "color":
        "#c0d5c2",  
        "head":
        "dead",
        "tail":
        "freckled",
        "version":
        "1"
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("Start Game:")
    snake_ai.reset()


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("End Game.\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
"""This function retrieves game_state, a dictionary containing all game data, 3 dicts and an int. Here's what it contains:

* game_state["turn"] returns an int, the current turn number of the game (like the current tick number)

* game_state["you"] returns the dictionary of data for our snake (a Battlesnake object), like coordinates of each piece              of the body, its length etc... 
see the details in the docs: https://docs.battlesnake.com/api/objects/battlesnake

* game_state["board"] returns the dictionary of data regarding the game board (a Board object), like the height and width of         the board, the coordinates of food, and the Battlesnake-objects of all snakes in the game etc... 
see the details in the docs: https://docs.battlesnake.com/api/objects/board

* game_state["game"] returns the dictionary of data regarding the game (a Game object), like the id of the game, the ruleset of the game (also a dict, it contains a dict "settings" about more specific set rules for this game), and the timeout time for the game (accessed as game_state["game"][timeout]) etc... 
see the details in the docs - Game object: https://docs.battlesnake.com/api/objects/game
                                    
Example of Move Request (the game_state received by move()) and Move Response (move()'s return value, new move dict):                 
https://docs.battlesnake.com/api/example-move"""

def move(game_state: typing.Dict) -> typing.Dict:
    # Snake AI
    snake_ai.update_state(game_state)

    next_move = snake_ai.get_Next_Move()
    
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move, "shout": "I'm a snake! Hissss!"}

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})

