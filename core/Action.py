import core.GameState as GameState
import core.utils as utils
import random

rankToInd = {GameState.Rank.ACE : 0, GameState.Rank.TWO : 1,
                     GameState.Rank.THREE : 2, GameState.Rank.FOUR : 3,
                     GameState.Rank.FIVE : 4, GameState.Rank.SIX : 5,
                     GameState.Rank.SEVEN : 6, GameState.Rank.EIGHT : 7,
                     GameState.Rank.NINE : 8, GameState.Rank.TEN : 9,
                     GameState.Rank.JACK : 10, GameState.Rank.QUEEN : 11,
                     GameState.Rank.KING : 12}
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
        hand = deck[left:right]
        for i in range(1, len(hand)):
            j = i
            while j > 0 and rankToInd[hand[j - 1].rank] > rankToInd[hand[j].rank]:
                temp = hand[j]
                hand[j] = hand[j - 1]
                hand[j - 1] = temp
                j -= 1
        player = GameState.PlayerState(id=i, hand=tuple(hand))
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
    #simple removal will never unsort
    newPlayerState = utils.updateSinglePlayer(players=state.players, player_id=state.current_player, newHand=tuple(newHand))


    claim = GameState.Claim(rank=state.current_rank, quantity=len(cards))

    truth = True
    newPile = state.pile
    # check for truth. also insert ordered into pile
    for card in cards:
        if card.rank != state.current_rank:
            truth = False
        newPile = utils.insertCardPile(newPile, card)

    return utils.updateState(state,
                             players=newPlayerState,
                             pile=newPile,
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
        hand = state.getPlayerById(state.current_player).hand
        i = 0
        j = 0
        newHand = []
        while i < len(hand) and j < len(state.pile):
            if rankToInd[hand[i].rank] < rankToInd[state.pile[j].rank]:
                newHand.append(hand[i])
                i += 1
            else:
                newHand.append(state.pile[j])
                j += 1

        if j < len(state.pile):
            newHand.append(state.pile[j:])
        if i < len(hand):
            newHand.append(hand[i:])

        newPlayerState = utils.updateSinglePlayer(
            players=state.players,
            player_id=state.current_player,
            newHand=tuple(newHand))
        newId = (state.current_player + 1) % state.playerCount
    else:
        #liar takes pile!
        hand = state.getPlayerById(state.last_actor).hand
        i = 0
        j = 0
        newHand = []
        while i < len(hand) and j < len(state.pile):
            if rankToInd[hand[i].rank] < rankToInd[state.pile[j].rank]:
                newHand.append(hand[i])
                i += 1
            else:
                newHand.append(state.pile[j])
                j += 1

        if j < len(state.pile):
            newHand.append(state.pile[j:])
        if i < len(hand):
            newHand.append(hand[i:])
        newPlayerState = utils.updateSinglePlayer(
            players=state.players,
            player_id=state.last_actor,
            newHand=tuple(newHand))
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
        prev_pile_size=len(state.pile),
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
        prev_pile_size=len(state.pile),
        winner=state.winner,
        turn_number=state.turn_number + 1,
    )
