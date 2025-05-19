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

# IMPORTANT INFO: When creating the battlesnakes, grab the server URL and add "snakeA" to the end of the URL when creating Snake A, such that the URL ends with "/snakeA". Similar for Snake D, the URL should end with "/snakeD". 

import typing
from SnakeAI import SnakeAI
from flask import Flask, request, jsonify #For collaboration between snakes

snake_a_ai = SnakeAI() # Create an instance of the SnakeAI class for Snake A
snake_d_ai = SnakeAI() # Create an instance of the SnakeAI class for Snake D

app = Flask(__name__) #Start web server in this file
shared_data = {}  # Shared data in dictionary for coordination between snakes

#Just testing the server
@app.route("/")
def server_root():
    return "Server is running!"

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
@app.route("/snakeA", methods=["GET"])
def info_snakeA() -> typing.Dict:
    print("INFO")
    return {
        "apiversion":
        "1",
        "Josepi The Moustache & Ram The Ham":
        "Johnny Aggro",
        "color":
        "#f20f29",  
        "head":
        "dead",
        "tail":
        "freckled",
        "version":
        "1"
    }
@app.route("/snakeD", methods=["GET"])
def info_snakeD() -> typing.Dict:
    print("INFO")
    return {
        "apiversion":
        "1",
        "Josepi The Moustache & Ram The Ham":
        "Johnny Defendo",
        "color":
        "#2417bd",  
        "head":
        "dead",
        "tail":
        "freckled",
        "version":
        "1"
    }

# start is called when your Battlesnake begins a game
@app.route("/snakeA/start", methods=["POST"])
def start_snakeA():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json() #Not used atm
    print("Start Game:")
    snake_a_ai.reset
    return {"status": "ok"} #Flask gets upset if this line is omitted
@app.route("/snakeD/start", methods=["POST"])
def start_snakeD():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json() #Not used atm
    print("Start Game:")
    snake_d_ai.reset()
    return {"status": "ok"}

# end is called when your Battlesnake finishes a game
@app.route("/snakeA/end", methods=["POST"])
@app.route("/snakeD/end", methods=["POST"])
def end():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json() #Not used atm
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

@app.route("/snakeA/move", methods=["POST"])
def move_snakeA():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json()
    
    # Snake AI
    snake_a_ai.update_state(game_state)

    next_move = snake_a_ai.get_Next_Move()
    
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move, "shout": "I'm snake A! Hissss!"}

@app.route("/snakeD/move", methods=["POST"])
def move_snakeD():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json()
    
    # Snake AI
    snake_d_ai.update_state(game_state)

    next_move = snake_d_ai.get_Next_Move()

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move, "shout": "I'm snake D! Hissss!"}

# Start server when `python main.py` is run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

