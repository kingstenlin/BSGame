from web.server.game_session import GameSession

class GameManager():
    def __init__(self):
        self.games = {}

    def create_game(self, id):
        session = GameSession()
        self.games[id] = session
        return session

    def get_game(self, id):
        return self.games.get(id)