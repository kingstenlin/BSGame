from core import GameState
from core.BSEnv import BSEnv
from newWeb.server.player import Player

class GameSession:
    def __init__(self):
        self.env = BSEnv()
        self.players = {} # int id to websocket
        self.playerct = 0

    def add_player(self, websocket):
        for pid in range(3):
            if pid not in self.players:
                self.players[pid] = Player(websocket=websocket)
                self.playerct += 1
                return pid
        return None

    def add_bot(self, player_id, agent):
        if player_id in self.players:
            return False

        self.players[player_id] = Player(agent=agent)
        return True

    async def broadcast(self):
        for playerID, player in self.players.items():
            playerObs = GameState.observe(self.env.state, playerID)
            if not player.is_bot:
                websocket = player.websocket
                await websocket.send_json({
                    "type" : "playerObs",
                    "data" : playerObs.toDict()
                })

    async def process_bot(self):
        while True:
            playerID = self.env.agent_selection
            player = self.players[playerID]

            if not player.is_bot:
                return

            obs = self.env._get_obs(playerID)
            mask = self.env._get_action_mask(playerID)
            action, _, _ = player.agent.act(obs, mask)

            self.env.step(action)

            await self.broadcast()