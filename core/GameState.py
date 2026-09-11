from dataclasses import dataclass
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, Optional

# TODO: implement history

class Rank(Enum):
    ACE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    TEN = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()

    def next(self):
        members = list(self.__class__)
        current_index = members.index(self)
        next_index = (current_index + 1) % len(members)
        return members[next_index]

    def prev(self):
        members = list(self.__class__)
        current_index = members.index(self)
        next_index = (current_index - 1) % len(members)
        return members[next_index]

class Suit(Enum):
    HEARTS = auto()
    DIAMONDS = auto()
    SPADES = auto()
    CLUBS = auto()

cardDict = {Rank.ACE : "A", Rank.TWO : "2", Rank.THREE : "3", Rank.FOUR : "4",
            Rank.FIVE : "5", Rank.SIX : "6", Rank.SEVEN : "7", Rank.EIGHT : "8",
            Rank.NINE : "9", Rank.TEN : "10", Rank.JACK : "J", Rank.QUEEN : "Q",
            Rank.KING : "K", Suit.HEARTS : "H", Suit.DIAMONDS : "D", Suit.SPADES : "S",
            Suit.CLUBS : "C"}

rankToInd = {Rank.ACE : 0, Rank.TWO : 1,
                     Rank.THREE : 2, Rank.FOUR : 3,
                     Rank.FIVE : 4, Rank.SIX : 5,
                     Rank.SEVEN : 6, Rank.EIGHT : 7,
                     Rank.NINE : 8, Rank.TEN : 9,
                     Rank.JACK : 10, Rank.QUEEN : 11,
                     Rank.KING : 12}

class Phase(Enum):
    DECLARE = auto()
    CHALLENGE = auto()

@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit

    def toDict(self):
        return {"rank": self.rank.name, "suit": self.suit.name}

@dataclass(frozen=True, slots=True)
class PlayerState:
    id: int
    hand: tuple[Card, ...]

    @property
    def size(self) -> int:
        return len(self.hand)

    def toDict(self):
        return {"id": self.id, "hand": self.hand}

@dataclass(frozen=True, slots=True)
class Claim:
    rank: Rank
    quantity: int

    def toDict(self):
        return {"rank": self.rank.name, "quantity": self.quantity}

@dataclass(frozen=True, slots=True)
class GameState:
    """
    land of dreams. 10 attributes
    """
    players: tuple[PlayerState, ...] # let them be sorted now
    pile: tuple[Card, ...] # also maintain sorted
    current_player: int
    current_phase: Phase
    current_claim: Optional[Claim] = None
    current_rank: Optional[Rank] = Rank.ACE
    last_actor: Optional[int] = None
    last_truth: Optional[bool] = None #was the last claim true?
    prev_pile_size: Optional[int] = 0
    winner: Optional[int] = None
    turn_number: int = 0


    @property
    def playerCount(self) -> int:
        return len(self.players)

    def getPlayerById(self, id: int) -> PlayerState:
        return self.players[id]

    def validate(self) -> None:
        """
        Defensive invariant checks.

        Useful during development/debugging.
        You can disable or strip these later if desired.
        """

        if not self.players:
            raise ValueError("Game must contain at least one player.")

        if self.current_player < 0 or self.current_player >= len(self.players):
            raise ValueError("Invalid current_player index.")

        # Ensure no duplicated cards exist.
        all_cards = []

        for player in self.players:
            all_cards.extend(player.hand)

        all_cards.extend(self.pile)

        if len(all_cards) != len(set(all_cards)):
            raise ValueError("Duplicate cards detected in state.")

    def printHand(self, playerId: int) -> None:
        print(f"Player {playerId}'s hand:")
        strings = []
        for card in self.players[playerId].hand:
            strings.append(cardDict[card.rank] + cardDict[card.suit])
        print(", ".join(strings))

    def printGame(self) -> None:
        for i in range(self.playerCount):
            self.printHand(i)

        print("Pile:")
        strings = []
        for card in self.pile:
            strings.append(cardDict[card.rank] + cardDict[card.suit])
        print(", ".join(strings))

@dataclass(frozen=True, slots=True)
class PlayerObservation:
    player_id: int
    player_hand: tuple[Card, ...]

    pile_size: int
    current_player: int
    phase: Phase

    last_actor: int

    hand_sizes: tuple[int, ...]
    turn_number: int

    current_claim: Optional[Claim] = None
    current_rank: Optional[Rank] = None

    historyQueue: tuple[int, ...] = ()

    def toDict(self):
        return {"player_id": self.player_id,
                "player_hand":[card.toDict() for card in self.player_hand],
                "pile_size": self.pile_size,
                "current_player": self.current_player,
                "phase": self.phase.name,
                "last_actor": self.last_actor,
                "hand_sizes": self.hand_sizes,
                "turn_number": self.turn_number,
                "current_claim": self.current_claim.toDict() if self.current_claim else None,
                "current_rank": self.current_rank.name if self.current_rank else None,
                "historyQueue": self.historyQueue}


def observe(state: GameState, player_id: int) -> PlayerObservation:
    player = state.getPlayerById(player_id)
    return PlayerObservation(
        player_id=player_id,
        player_hand=player.hand,
        pile_size=len(state.pile),
        current_player=state.current_player,
        phase=state.current_phase,
        last_actor=state.last_actor,
        hand_sizes=tuple(p.size for p in state.players),
        turn_number=state.turn_number,
        current_claim=state.current_claim,
        current_rank=state.current_rank,
    )

