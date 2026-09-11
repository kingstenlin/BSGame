# TODO: handle disconnect

from fastapi import FastAPI, WebSocket
from web.server.game_manager import GameManager

import json

from core import BSEnv, Action, GameState
from RL import agents
from RL.utils import load_policy_agent

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
    # create session
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
    # on join, give state
    playerObs = GameState.observe(session.env.state, playerID)
    data = {"type": "playerObs",
            "data": playerObs.toDict()}
    await websocket.send_json(json.dumps(data))

    while True:
        message = await websocket.receive_json()

        # info/debugging ---------------------------------------------------------------
        # const ws = new WebSocket("ws://localhost:8000/ws/test");
        if message["type"] == "debug":
            data = message["data"]
            if data == "getPlayer":
                await websocket.send_json({"type": "debug",
                                           "data": playerID})
        # make bots ---------------------------------------------------------------

        # ws.send(JSON.stringify({
        #     type: "add_bot",
        #     data: {player_id : 1, agent : "john"}
        # }));

        if message["type"] == "add_bot":
            player_id = message["data"]["player_id"]
            agent_name = message["data"]["agent"]
            if agent_name == "policy":
                ag = load_policy_agent("./RL/checkpointsPPO/policy_0100000.pt", "mps")
            else:
                ag = agents.RandomAgent()
            session.add_bot(player_id, ag)

        # action area ---------------------------------------------------------------

        # ws.send(JSON.stringify({
        #     type: "action",
        #     data: 2
        # }));

        if message["type"] == "action":
            validActions = session.env._get_action_mask(playerID)

            action = message["data"]
            if validActions[action]:
                session.env.step(action)
                await session.broadcast()
                await session.process_bot()
            else:
                data = {"type" : "error",
                        "data" : "Invalid action"}
                await websocket.send_json(data)