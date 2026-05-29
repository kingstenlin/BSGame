import numpy as np
import engineTest, GameState, BSEnv

currRank = GameState.Rank.ACE
tw = engineTest.make_card(GameState.Rank.TWO)
th = engineTest.make_card(GameState.Rank.THREE)
fi = engineTest.make_card(GameState.Rank.FIVE)
hand = [fi, th, tw]


def relRank(card: GameState.Card) -> int:
    return (BSEnv.rankToInd[card.rank] - BSEnv.rankToInd[currRank]) % 13



print(1 == True)