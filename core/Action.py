import core.GameState as GameState
import core.utils as utils
import random

#TODO: support > 1 deck. dependencies are everywhere

#these should all return GameStates

def initializeGame(player_ct, decks = 1, seed = 42):
    #make a hand for each player by shuffling the deck and drawing
    deck = [GameState.Card(rank, suit) for _ in range(decks) for rank in GameState.Rank for suit in GameState.Suit]

    random.seed(seed)
    random.shuffle(deck)

    cards_per = len(deck) // player_ct
    remainder = len(deck) % player_ct
    #two pointer system to draw hands and accomodate leftovers
    left = 0
    right = cards_per
    players = []
    for i in range(player_ct):
        if remainder > 0:
            right += 1
            remainder -= 1
        player = GameState.PlayerState(id=i, hand=tuple(deck[left:right]))
        players.append(player)
        left = right
        right += cards_per

    return GameState.GameState(
        players=tuple(players),
        pile=(),
        current_player=0,
        current_phase=GameState.Phase.DECLARE,
        current_claim=None,
        current_rank=GameState.Rank.ACE,
        last_actor = None,
        last_truth = None,
        winner = None,
        turn_number = 0
    )

def playCards(state:GameState.GameState, cards: tuple[GameState.Card, ...]):
    #does not support deck > 1

    #ensure we are in correct state
    if state.current_phase != GameState.Phase.DECLARE:
        raise Exception("Out of phase")

    # ensure action is trying to play at least one card
    if len(cards) < 1:
        raise Exception("Must play at least one card")


    # the only time anyone can win is if they play a card and the challenge
    # doesn't happen/is unsuccessful. hence, check previous player's hand size
    # to check winner
    prev = (state.current_player - 1) % state.playerCount
    if len(state.getPlayerById(prev).hand) == 0:
        return utils.updateState(state, winner=prev)


    #modify the player's hand to actually play them.
    activeHand = state.players[state.current_player].hand
    newHand = []
    c = 0
    for activeCard in activeHand:
        if activeCard not in cards:
            newHand.append(activeCard)
        else:
            c += 1

    if c != len(cards):
        raise Exception("Cards failed to play: invalid selection")

    newPlayerState = utils.updateSinglePlayer(players=state.players, player_id=state.current_player, newHand=tuple(newHand))


    claim = GameState.Claim(rank=state.current_rank, quantity=len(cards))

    #record if the claim was true
    truth = True
    for card in cards:
        if card.rank != state.current_rank:
            truth = False
    return utils.updateState(state,
                             players=newPlayerState,
                             pile=state.pile + tuple(cards),
                             current_player=(state.current_player + 1) % state.playerCount,
                             current_phase = GameState.Phase.CHALLENGE,
                             current_claim = claim,
                             last_actor = state.current_player,
                             last_truth = truth,
                             turn_number = state.turn_number + 1)


def challenge(state:GameState.GameState):
    if state.current_phase != GameState.Phase.CHALLENGE:
        raise Exception("Out of phase")

    #check if the challenge is successful, and act accordingly
    #the next player is the one after the one who picks up the cards

    if state.last_truth:
        # if it was true and the prev is done, game is over
        prev = (state.current_player - 1) % state.playerCount
        if len(state.getPlayerById(prev).hand) == 0:
            return utils.updateState(state, winner=prev)

        #challenger takes pile!
        newPlayerState = utils.updateSinglePlayer(
            players=state.players,
            player_id=state.current_player,
            newHand=state.getPlayerById(state.current_player).hand + state.pile)
        newId = (state.current_player + 1) % state.playerCount
    else:
        #liar takes pile!
        newPlayerState = utils.updateSinglePlayer(
            players=state.players,
            player_id=state.last_actor,
            newHand=state.getPlayerById(state.last_actor).hand + state.pile)
        newId = (state.last_actor + 1) % state.playerCount


    return GameState.GameState(
        players=newPlayerState,
        pile=(),
        current_player=newId,
        current_phase=GameState.Phase.DECLARE,
        current_claim=None,
        current_rank=state.current_rank.next(), #todo: optional rank reset
        last_actor=state.last_actor,  # maintain for reward calculating
        last_truth=state.last_truth,
        winner=state.winner,
        turn_number = state.turn_number + 1,
    )


def passChallenge(state:GameState.GameState):
    if state.current_phase != GameState.Phase.CHALLENGE:
        raise Exception("Out of phase")

    return GameState.GameState(
        players=state.players,
        pile=state.pile,
        current_player=state.current_player, # doesn't move since after you play after challenge
        current_phase=GameState.Phase.DECLARE,
        current_claim=None,
        current_rank=state.current_rank.next(),
        last_actor=state.last_actor, # maintain for reward calculating
        last_truth=state.last_truth,
        winner=state.winner,
        turn_number=state.turn_number + 1,
    )
