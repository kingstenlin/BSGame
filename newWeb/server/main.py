from fastapi import FastAPI, WebSocket
from newWeb.server.game_manager import GameManager

import json

from core import BSEnv, Action, GameState

#uvicorn newWeb.server.main:app --reload
app = FastAPI()

manager = GameManager()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws/{id}")
async def websocket_endpoint(websocket: WebSocket, id: str):
    await websocket.accept()
    session = manager.get_game(id)
    if session is None:
        session = manager.create_game(id)

    if session.playerct == 0:
        session.env.reset()

    playerID = session.add_player(websocket)

    if playerID is None:
        await websocket.send_json({"type": "error",
                                  "data" : "Full lobby"})
        return
    else:
        await websocket.send_json({"type": "player_assigned",
                                   "data": playerID})

    while True:
        # give observation to the player
        playerObs = GameState.observe(session.env.state, playerID)
        data = {"type": "playerObs",
                "data": playerObs.toDict()}
        await websocket.send_json(json.dumps(data))


        message = await websocket.receive_json()
        if message["type"] == "action":
            validActions = session.env._get_action_mask(playerID)

            action = message["data"]
            if validActions[action]:
                session.env.step(action)
                playerObs = GameState.observe(session.env.state, playerID)
                data = {"type" : "playerObs",
                        "data" : playerObs.toDict()}
            else:
                data = {"type" : "error",
                        "data" : "Invalid action"}
            await websocket.send_json(json.dumps(data))
