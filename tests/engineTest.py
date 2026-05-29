"""
pytest engineTest.py -v

Test suite for the BS game engine (Action.py + GameState.py).

Coverage targets:
    - Initialization invariants
    - playCards transitions
    - passChallenge transitions
    - challenge transitions (honest and bluff)
    - Win condition detection (both paths)
    - Cross-cutting card count invariant
"""

import pytest
from GameState import (
    Card, Rank, Suit, Phase, Claim,
    GameState, PlayerState, observe
)
import Action
from utils import *

# -----------------------------------------------------------------------
# 1. Initialization
# -----------------------------------------------------------------------

class TestInitialization:

    def test_total_card_count(self):
        state = Action.initializeGame(player_ct=3)
        assert total_cards(state) == 52

    def test_no_duplicate_cards(self):
        state = Action.initializeGame(player_ct=3)
        all_cards = [c for p in state.players for c in p.hand]
        assert len(all_cards) == len(set(all_cards))

    def test_starting_phase_is_declare(self):
        state = Action.initializeGame(player_ct=3)
        assert state.current_phase == Phase.DECLARE

    def test_starting_rank_is_ace(self):
        state = Action.initializeGame(player_ct=3)
        assert state.current_rank == Rank.ACE

    def test_starting_player_is_zero(self):
        state = Action.initializeGame(player_ct=3)
        assert state.current_player == 0

    def test_hands_are_nonempty(self):
        state = Action.initializeGame(player_ct=3)
        for player in state.players:
            assert len(player.hand) > 0

    def test_hand_sizes_differ_by_at_most_one(self):
        """52 cards into 3 players: one player gets 18, two get 17."""
        state = Action.initializeGame(player_ct=3)
        sizes = [p.size for p in state.players]
        assert max(sizes) - min(sizes) <= 1

    def test_validate_passes_on_fresh_state(self):
        state = Action.initializeGame(player_ct=3)
        state.validate()  # should not raise


# -----------------------------------------------------------------------
# 2. playCards
# -----------------------------------------------------------------------

class TestPlayCards:

    def test_cards_removed_from_hand(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]  # 4 aces
        other = [make_card(Rank.TWO, Suit.HEARTS)]
        state = make_state(hands=[aces + other, [make_card(Rank.THREE, Suit.CLUBS)], [make_card(Rank.FOUR, Suit.CLUBS)]])
        to_play = (aces[0], aces[1])
        new_state = Action.playCards(state, to_play)
        new_hand = new_state.getPlayerById(0).hand
        assert aces[0] not in new_hand
        assert aces[1] not in new_hand
        assert aces[2] in new_hand
        assert aces[3] in new_hand

    def test_cards_added_to_pile(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        to_play = (aces[0],)
        new_state = Action.playCards(state, to_play)
        assert aces[0] in new_state.pile

    def test_phase_transitions_to_challenge(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        new_state = Action.playCards(state, (aces[0],))
        assert new_state.current_phase == Phase.CHALLENGE

    def test_current_player_advances(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        new_state = Action.playCards(state, (aces[0],))
        assert new_state.current_player == 1

    def test_last_actor_set_correctly(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        new_state = Action.playCards(state, (aces[0],))
        assert new_state.last_actor == 0

    def test_honest_play_sets_last_truth_true(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        new_state = Action.playCards(state, (aces[0],))
        assert new_state.last_truth is True

    def test_bluff_play_sets_last_truth_false(self):
        # Play a non-ACE card when rank is ACE
        bluff_card = make_card(Rank.TWO, Suit.HEARTS)
        state = make_state(
            hands=[[bluff_card, make_card(Rank.THREE, Suit.HEARTS)],
                   [make_card(Rank.FOUR, Suit.HEARTS)],
                   [make_card(Rank.FIVE, Suit.HEARTS)]],
            current_rank=Rank.ACE
        )
        new_state = Action.playCards(state, (bluff_card,))
        assert new_state.last_truth is False

    def test_rank_does_not_advance_in_play_cards(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        new_state = Action.playCards(state, (aces[0],))
        assert new_state.current_rank == Rank.ACE  # not TWO

    def test_card_count_preserved(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        others = [make_card(r, Suit.CLUBS) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                  Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
                  Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING]]
        # Build a 3-player state with exactly 16 total cards
        state = make_state(
            hands=[aces, others[:6], others[6:]],
        )
        expected = total_cards(state)
        new_state = Action.playCards(state, (aces[0],))
        assert_card_count_invariant(new_state, expected)

    def test_cannot_play_cards_not_in_hand(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces[:2], [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        foreign_card = make_card(Rank.KING, Suit.SPADES)
        with pytest.raises(Exception):
            Action.playCards(state, (foreign_card,))

    def test_cannot_play_zero_cards(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        with pytest.raises(Exception):
            Action.playCards(state, ())

    def test_wrong_phase_raises(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(
            hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]],
            phase=Phase.CHALLENGE
        )
        with pytest.raises(Exception):
            Action.playCards(state, (aces[0],))


# -----------------------------------------------------------------------
# 3. passChallenge
# -----------------------------------------------------------------------

class TestPassChallenge:

    def _state_after_play(self):
        """Helper: build a post-playCards state for challenge phase tests."""
        aces = [make_card(Rank.ACE, s) for s in Suit]
        p1 = [make_card(Rank.THREE, s) for s in Suit]
        p2 = [make_card(Rank.THREE, s) for s in Suit]
        state = make_state(hands=[aces, p1, p2], current_rank=Rank.ACE)
        return Action.playCards(state, (aces[0],))


    def test_rank_advances_exactly_once(self):
        """
        After bug fix: rank advances in passChallenge (not in playCards).
        Full turn: ACE → TWO, not ACE → THREE.
        """
        post_play = self._state_after_play()
        assert post_play.current_rank == Rank.ACE  # not yet advanced
        post_pass = Action.passChallenge(post_play)
        assert post_pass.current_rank == Rank.TWO  # advanced exactly once

    def test_current_player_unchanged(self):
        """Challenger passes, then plays — same player stays active."""
        post_play = self._state_after_play()
        challenger = post_play.current_player
        post_pass = Action.passChallenge(post_play)
        assert post_pass.current_player == challenger

    def test_phase_returns_to_declare(self):
        post_play = self._state_after_play()
        post_pass = Action.passChallenge(post_play)
        assert post_pass.current_phase == Phase.DECLARE

    def test_pile_preserved(self):
        post_play = self._state_after_play()
        post_pass = Action.passChallenge(post_play)
        assert len(post_pass.pile) == len(post_play.pile)

    def test_last_truth_maintained(self):
        """last_truth should maintain into the next DECLARE state."""
        post_play = self._state_after_play()
        post_pass = Action.passChallenge(post_play)
        assert post_pass.last_truth is not None

    def test_last_actor_maintained(self):
        post_play = self._state_after_play()
        post_pass = Action.passChallenge(post_play)
        assert post_pass.last_actor is not None

    def test_deep_last_truth_changed(self):

        """after a two turn, truth might change"""
        # in the test case, the first declaration is true
        post_play = self._state_after_play()
        assert post_play.last_truth == 1
        s1 = Action.passChallenge(post_play)
        assert s1.last_truth == 1
        # play a bunch of threes during a two turn
        s2 = Action.playCards(s1, s1.getPlayerById(s1.current_player).hand)
        assert s2.last_truth == 0

    def test_card_count_preserved(self):
        post_play = self._state_after_play()
        expected = total_cards(post_play)
        post_pass = Action.passChallenge(post_play)
        assert_card_count_invariant(post_pass, expected)

    def test_wrong_phase_raises(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        state = make_state(hands=[aces, [make_card(Rank.TWO, Suit.HEARTS)], [make_card(Rank.THREE, Suit.HEARTS)]])
        with pytest.raises(Exception):
            Action.passChallenge(state)


# -----------------------------------------------------------------------
# 4. challenge — honest play (challenger loses)
# -----------------------------------------------------------------------

class TestChallengeHonest:
    """last_truth = True: claimant was honest, challenger collects pile."""

    def _state_after_honest_play(self):
        aces = [make_card(Rank.ACE, s) for s in Suit]
        p1 = [make_card(Rank.TWO, s) for s in Suit]
        p2 = [make_card(Rank.THREE, s) for s in Suit]
        state = make_state(hands=[aces, p1, p2], current_rank=Rank.ACE)
        return Action.playCards(state, (aces[0],))  # honest: ace played on ace turn

    def test_challenger_collects_pile(self):
        post_play = self._state_after_honest_play()
        challenger_id = post_play.current_player
        pile_size = len(post_play.pile)
        post_challenge = Action.challenge(post_play)
        new_hand_size = post_challenge.getPlayerById(challenger_id).size
        # challenger's hand grew by the pile size
        original_size = post_play.getPlayerById(challenger_id).size
        assert new_hand_size == original_size + pile_size

    def test_pile_empties(self):
        post_play = self._state_after_honest_play()
        post_challenge = Action.challenge(post_play)
        assert len(post_challenge.pile) == 0

    def test_current_player_after_challenger(self):
        post_play = self._state_after_honest_play()
        challenger_id = post_play.current_player
        expected_next = (challenger_id + 1) % 3
        post_challenge = Action.challenge(post_play)
        assert post_challenge.current_player == expected_next

    def test_rank_advances_exactly_once(self):
        post_play = self._state_after_honest_play()
        assert post_play.current_rank == Rank.ACE
        post_challenge = Action.challenge(post_play)
        assert post_challenge.current_rank == Rank.TWO

    def test_claimant_hand_unchanged(self):
        post_play = self._state_after_honest_play()
        claimant_id = post_play.last_actor
        original_size = post_play.getPlayerById(claimant_id).size
        post_challenge = Action.challenge(post_play)
        assert post_challenge.getPlayerById(claimant_id).size == original_size

    def test_card_count_preserved(self):
        post_play = self._state_after_honest_play()
        expected = total_cards(post_play)
        post_challenge = Action.challenge(post_play)
        assert_card_count_invariant(post_challenge, expected)


# -----------------------------------------------------------------------
# 5. challenge — bluff (claimant loses)
# -----------------------------------------------------------------------

class TestChallengeBluff:
    """last_truth = False: claimant bluffed, claimant collects pile."""

    def _state_after_bluff_play(self):
        bluff_card = make_card(Rank.TWO, Suit.HEARTS)  # not an ace
        p0 = [bluff_card, make_card(Rank.THREE, Suit.HEARTS)]
        p1 = [make_card(Rank.FOUR, Suit.HEARTS), make_card(Rank.FIVE, Suit.HEARTS)]
        p2 = [make_card(Rank.SIX, Suit.HEARTS), make_card(Rank.SEVEN, Suit.HEARTS)]
        state = make_state(hands=[p0, p1, p2], current_rank=Rank.ACE)
        return Action.playCards(state, (bluff_card,))  # bluff: TWO played on ACE turn

    def test_bluffer_collects_pile(self):
        post_play = self._state_after_bluff_play()
        bluffer_id = post_play.last_actor
        pile_size = len(post_play.pile)
        original_size = post_play.getPlayerById(bluffer_id).size
        post_challenge = Action.challenge(post_play)
        assert post_challenge.getPlayerById(bluffer_id).size == original_size + pile_size

    def test_pile_empties(self):
        post_play = self._state_after_bluff_play()
        post_challenge = Action.challenge(post_play)
        assert len(post_challenge.pile) == 0

    def test_current_player_after_bluffer(self):
        post_play = self._state_after_bluff_play()
        bluffer_id = post_play.last_actor
        expected_next = (bluffer_id + 1) % 3
        post_challenge = Action.challenge(post_play)
        assert post_challenge.current_player == expected_next

    def test_challenger_hand_unchanged(self):
        post_play = self._state_after_bluff_play()
        challenger_id = post_play.current_player
        original_size = post_play.getPlayerById(challenger_id).size
        post_challenge = Action.challenge(post_play)
        assert post_challenge.getPlayerById(challenger_id).size == original_size

    def test_card_count_preserved(self):
        post_play = self._state_after_bluff_play()
        expected = total_cards(post_play)
        post_challenge = Action.challenge(post_play)
        assert_card_count_invariant(post_challenge, expected)


# -----------------------------------------------------------------------
# 6. Win conditions
# -----------------------------------------------------------------------

class TestWinConditions:

    def test_win_via_pass_winner_is_correct_player(self):
        """
        Player 0 plays their last card honestly.
        Player 1 passes the challenge.
        Next call to playCards should detect player 0 as winner.
        """
        last_ace = make_card(Rank.ACE, Suit.HEARTS)
        p0 = [last_ace]  # one card left
        p1 = [make_card(Rank.TWO, s) for s in Suit]
        p2 = [make_card(Rank.THREE, s) for s in Suit]
        state = make_state(hands=[p0, p1, p2], current_rank=Rank.ACE)

        post_play = Action.playCards(state, (last_ace,))
        assert len(post_play.getPlayerById(0).hand) == 0

        post_pass = Action.passChallenge(post_play)
        # Player 1 now plays — this triggers winner detection
        final_state = Action.playCards(post_pass, (p1[0],))

        assert final_state.winner == 0  # int, not True
        assert isinstance(final_state.winner, int)

    def test_win_via_honest_challenge_winner_is_correct_player(self):
        """
        Player 0 plays their last card honestly.
        Player 1 challenges (and loses — claim was honest).
        Winner should be detected inside challenge().
        """
        last_ace = make_card(Rank.ACE, Suit.HEARTS)
        p0 = [last_ace]
        p1 = [make_card(Rank.TWO, s) for s in Suit]
        p2 = [make_card(Rank.THREE, s) for s in Suit]
        state = make_state(hands=[p0, p1, p2], current_rank=Rank.ACE)

        post_play = Action.playCards(state, (last_ace,))
        assert len(post_play.getPlayerById(0).hand) == 0
        assert post_play.last_truth is True

        post_challenge = Action.challenge(post_play)
        assert post_challenge.winner == 0
        assert isinstance(post_challenge.winner, int)

    def test_bluff_then_challenge_no_winner(self):
        """
        Player 0 plays their last card as a bluff.
        Player 1 challenges successfully.
        Player 0 collects the pile back — no winner.
        """
        bluff_card = make_card(Rank.TWO, Suit.HEARTS)  # not an ace
        p0 = [bluff_card]
        p1 = [make_card(Rank.THREE, s) for s in Suit]
        p2 = [make_card(Rank.FOUR, s) for s in Suit]
        state = make_state(hands=[p0, p1, p2], current_rank=Rank.ACE)

        post_play = Action.playCards(state, (bluff_card,))
        post_challenge = Action.challenge(post_play)

        assert post_challenge.winner is None
        assert post_challenge.getPlayerById(0).size > 0


# -----------------------------------------------------------------------
# 7. Cross-cutting invariant: card count across a full simulated game
# -----------------------------------------------------------------------

class TestCardCountInvariant:

    def test_card_count_preserved_across_full_game(self):
        """
        Run a full game with a fixed strategy (always honest, never challenge)
        and assert 52 cards exist at every state transition.

        This is an integration check — if any transition gains or loses cards,
        this will catch it.
        """
        state = Action.initializeGame(player_ct=3, seed=0)
        expected = total_cards(state)

        for _ in range(500):  # cap at 500 transitions
            assert_card_count_invariant(state, expected)
            state.validate()

            if state.winner is not None:
                break

            if state.current_phase == Phase.DECLARE:
                hand = state.getPlayerById(state.current_player).hand
                state = Action.playCards(state, (hand[0],))
            elif state.current_phase == Phase.CHALLENGE:
                state = Action.passChallenge(state)