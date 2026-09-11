class Player:
    def __init__(self, websocket = None, agent = None):
        self.websocket = websocket
        self.agent = agent

    @property
    def is_bot(self):
        return self.agent is not None