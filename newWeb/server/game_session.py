from pettingzoo.butterfly.knights_archers_zombies.src.players import Player

from core.BSEnv import BSEnv


class GameSession:
    def __init__(self):
        self.env = BSEnv()
        self.players = {} # int id to websocket
        self.playerct = 0

    def add_player(self, websocket):
        for pid in range(3):
            if pid not in self.players:
                self.players[pid] = websocket
                self.playerct += 1
                return pid
        return None