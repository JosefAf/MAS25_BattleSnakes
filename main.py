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
from AttackerAI import AttackerAI
from DefenderAI import DefenderAI
from flask import Flask, request, jsonify #For collaboration between snakes
import time

snake_a_ai = AttackerAI() # Create an instance of the SnakeAI class for Snake A
snake_d_ai = DefenderAI() # Create an instance of the SnakeAI class for Snake D

app = Flask(__name__) #Start web server in this file
attacker_data = { #attacker data for defender to read and use when planning its next move
    "ID" : None, #Each snake's unique IDs is set by the engine and must be read by us
    "turn" : None, #for making sure available data is for the current turn
    "next_move" : None, #attacker's next move
    "closest_food_pos" : None #reserve food closest to snake A to snake A? Unused atm
}
defender_ID = None

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
        "#e77aeb",  
        "head":
        "dead",
        "tail":
        "sharp",
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
        "#e77aeb",  
        "head":
        "dead",
        "tail":
        "round-bum",
        "version":
        "1"
    }

# start is called when your Battlesnake begins a game
@app.route("/snakeA/start", methods=["POST"])
def start_snakeA():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json()
    #Read attacker's ID and write to attacker_data
    attacker_data["ID"] = game_state["you"]["id"]
    print("Start Game:")
    snake_a_ai.reset
    return {"status": "ok"} #Flask gets upset if this line is omitted
@app.route("/snakeD/start", methods=["POST"])
def start_snakeD():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json() #Not used atm
    defender_ID = game_state["you"]["id"]
    print("Start Game:")
    snake_d_ai.reset()
    return {"status": "ok"} #Flask gets upset if this line is omitted

# end is called when your Battlesnake finishes a game
@app.route("/snakeA/end", methods=["POST"])
@app.route("/snakeD/end", methods=["POST"])
def end():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json() #Not used atm
    print("End Game.\n")
    return {"status": "ok"} #Flask gets upset if this line is omitted


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
    
    snake_a_ai.update_state(game_state, defender_ID)
    next_move = snake_a_ai.get_Next_Move()
    
    #Update attacker_data for defender
    current_turn = game_state["turn"]
    attacker_data["turn"] = current_turn
    attacker_data["next_move"] = next_move
    #TODO attacker_data["closest_food_pos"] = snake_a_ai.get_Closest_Food_Pos() #method not implemented yet
    
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move, "shout": "I'm snake A! Hissss!"}

@app.route("/snakeD/move", methods=["POST"])
def move_snakeD():
    #Retrieve game_state like this when using Flask:
    game_state = request.get_json()

    #Use attacker snake's ID in attacker_data to check if attacker is still alive before deciding to wait for attacker_data to be updated for this turn.
    attacker_id = attacker_data["ID"]
    alive_snakes = game_state["board"]["snakes"]
    attacker_alive = any(snake["id"] == attacker_id for snake in alive_snakes)

    if attacker_alive:
        current_turn = game_state["turn"]
        # Wait (a few ms at a time) until attacker move for this turn is available
        timeout = 0.2  # Max 200 ms wait
        start_time = time.time()
        while time.time() - start_time < timeout:
            #Check if attacker_data has been updated for this turn
            if attacker_data["turn"] == current_turn:
                break
            else:
                time.sleep(0.005)  # Wait 5 ms

        timeout_reached = False
        if time.time() - start_time >= timeout:
            timeout_reached = True

        #Reaching this point means attacker alive and we have attacker data for this turn or timeout reached
        if timeout_reached:
            snake_d_ai.update_state(game_state, None)
        else:
            snake_d_ai.update_state(game_state, attacker_data)

        next_move = snake_d_ai.get_Next_Move()
    
        print(f"MOVE {game_state['turn']}: {next_move}")
        return {"move": next_move, "shout": "I'm snake D! Hissss!"}

    else: #attacker dead, move on without attacker_data
        snake_d_ai.update_state(game_state, None)
        next_move = snake_d_ai.get_Next_Move()
    
        print(f"MOVE {game_state['turn']}: {next_move}")
        return {"move": next_move, "shout": "I'm snake D! Hissss!"}

# Start server when `python main.py` is run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

