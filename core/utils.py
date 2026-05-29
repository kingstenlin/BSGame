import core.GameState as GameState
from typing import Optional
def updateSinglePlayer(players: tuple[GameState.PlayerState, ...],
                       player_id: int,
                       newHand: tuple[GameState.Card, ...]) -> tuple[GameState.PlayerState, ...]:
    newPlayers = []

    for player in players:
        if player.id == player_id:
            newPlayers.append(GameState.PlayerState(player_id, newHand))
        else:
            newPlayers.append(player)

    return tuple(newPlayers)

def updateState(state: GameState.GameState,
                players: tuple[GameState.PlayerState, ...] = None,
                pile: tuple[GameState.Card, ...] = None,
                current_player: int = None,
                current_phase: GameState.Phase = None,
                current_claim: GameState.Claim = None,
                current_rank: GameState.Rank = None,
                last_actor: int = None,
                last_truth: int = None,
                winner: int = None,
                turn_number: int = None
                ):
    return GameState.GameState(
        players = players if (players is not None) else state.players,
        pile = pile if (pile is not None) else state.pile,
        current_player = current_player if (current_player is not None) else state.current_player,
        current_phase = current_phase if (current_phase is not None) else state.current_phase,
        current_claim = current_claim if (current_claim is not None) else state.current_claim,
        current_rank = current_rank if (current_rank is not None) else state.current_rank,
        last_actor = last_actor if (last_actor is not None) else state.last_actor,
        last_truth = last_truth if (last_truth is not None) else state.last_truth,
        winner = winner if (winner is not None) else state.winner,
        turn_number = turn_number if (turn_number is not None) else state.turn_number
    )

def make_card(rank: GameState.Rank, suit: GameState.Suit = GameState.Suit.HEARTS) -> GameState.Card:
    return GameState.Card(rank=rank, suit=suit)


def make_player(id: int, cards: list[GameState.Card]) -> GameState.PlayerState:
    return GameState.PlayerState(id=id, hand=tuple(cards))


def make_state(
    hands: list[list[GameState.Card]],
    pile: list[GameState.Card] = None,
    current_player: int = 0,
    phase: GameState.Phase = GameState.Phase.DECLARE,
    current_rank: GameState.Rank = GameState.Rank.ACE,
    current_claim: GameState.Claim = None,
    last_actor: int = None,
    last_truth: bool = None,
    winner: int = None,
    turn_number: int = 0,
) -> GameState:

    players = tuple(make_player(i, hands[i]) for i in range(len(hands)))
    return GameState.GameState(
        players=players,
        pile=tuple(pile or []),
        current_player=current_player,
        current_phase=phase,
        current_rank=current_rank,
        current_claim=current_claim,
        last_actor=last_actor,
        last_truth=last_truth,
        winner=winner,
        turn_number=turn_number,
    )


def total_cards(state: GameState) -> int:
    """Count all cards in play: hands + pile."""
    return sum(len(p.hand) for p in state.players) + len(state.pile)


def assert_card_count_invariant(state: GameState, expected: int = 52):
    """No cards should appear or disappear between transitions."""
    assert total_cards(state) == expected, (
        f"Card count violation: expected {expected}, got {total_cards(state)}"
    )