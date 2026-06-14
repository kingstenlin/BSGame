import core.Action as Action
import core.utils as utils
import core.GameState as GameState
import pettingzoo
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


NUM_DECKS = 1
NUM_PLAYERS = 3
rankToInd = {GameState.Rank.ACE : 0, GameState.Rank.TWO : 1,
                     GameState.Rank.THREE : 2, GameState.Rank.FOUR : 3,
                     GameState.Rank.FIVE : 4, GameState.Rank.SIX : 5,
                     GameState.Rank.SEVEN : 6, GameState.Rank.EIGHT : 7,
                     GameState.Rank.NINE : 8, GameState.Rank.TEN : 9,
                     GameState.Rank.JACK : 10, GameState.Rank.QUEEN : 11,
                     GameState.Rank.KING : 12}

rankToCyclicalSin = {r : np.sin(2 * np.pi * rankToInd[r] / 13) / 2 + 0.5
                  for r in list(GameState.Rank.__members__.values())}
rankToCyclicalCos = {r : np.cos(2 * np.pi * rankToInd[r] / 13) / 2 + 0.5
                  for r in list(GameState.Rank.__members__.values())}
# Observation vector layout:
# [0:13]: counts of each rank, normalized by 4
# [13, 14]: cyclic encoding of current rank
# [15]: pile size, normalized by deck size
# [16]: claim quantity, normalized by 4 (0 if no claim)
# [17]: current phase (0 for DECLARE, 1 for CHALLENGE)
# Next NUM_PLAYERS: hand sizes, normalized by deck size
# TODO: this shall now be ordered from perspective of the agent as curr, next, prev

OBS_DIM = 29 + NUM_PLAYERS

# Action space:
# 0 declare 1 card, 0 of which are honest
# 1 declare 1 card, 1 of which are honest
# 2 declare 2 cards, 0 of which are honest
# 3 declare 2 cards, 1 of which are honest
# 4 d 2 c, 2 o w a h
# 5 3, 0
# 6 3, 1
# 7 3, 2
# 8 3, 3
# 9 4, 0
# 10 4, 1
# 11 4, 2
# 12 4, 3
# 13 4, 4
# 14 Challenge
# 15 Pass challenge

declareActions = [
    (qty, honest)
    for qty in range(1, 5)
    for honest in range(0, qty + 1)
]  # 14 entries, index = action integer

NUM_ACTIONS = 16

class BSEnv(AECEnv):
    metadata = {"render_modes" : ["human"], "name" : "BS_v1"}

    def __init__(self, renderMode = None, max_iter = 100000):
        super().__init__()
        self.possible_agents = [r for r in range(3)]
        self.state: Optional[GameState.GameState] = None
        self._rng = np.random.default_rng(seed=42)
        self.render_mode = renderMode
        self.max_iter = max_iter
        self._action_space = {a : spaces.Discrete(NUM_ACTIONS) for a in self.possible_agents}
        self._observation_space = {a : spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        ) for a in self.possible_agents}

    def action_space(self, agent):
        return self._action_space[agent]

    def observation_space(self, agent):
        return self._observation_space[agent]

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        return self.state.printGame()


    def close(self):
        pass

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        game_seed = int(self._rng.integers(0, 2 ** 31))

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.state = Action.initializeGame(player_ct=NUM_PLAYERS, seed=game_seed)
        self.observations = {agent: self._get_obs(agent) for agent in self.agents}
        self.agent_selection = self.state.current_player
        self.num_moves = 0
        self.infos = {
            agent: {
                "action_mask": self._get_action_mask(agent)
            }
            for agent in self.agents
        }

    def step(self, action: int):
        """
                step(action) takes in an action for the current agent (specified by
                agent_selection) and needs to update
                - rewards
                - _cumulative_rewards (accumulating the rewards)
                - terminations
                - truncations
                - infos
                - agent_selection (to the next agent)
                And any internal state used by observe() or render()
                """
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        # reset rewards
        self._clear_rewards()


        # stores action of current agent
        self.state = self._apply_action(action)

        # update terminations.
        # this is also where all the win loss reward lives
        if self.state.winner is not None:
            for a in self.agents:
                self.terminations[a] = True
                if a == self.state.winner:
                    self.rewards[a] = 1
                else:
                    self.rewards[a] = -0.5


        # handle the general reward for current agent
        self._compute_reward(acting_player=agent, terminated=self.terminations[agent], action=action)

        # TODO: handle the retroactive reward for a successful or failed bluff

        self.num_moves += 1

        if self.num_moves >= self.max_iter:
            for a in self.agents:
                self.truncations[a] = True
                # TODO: consider truncation penalty?

        # observe the current state for all agents
        for i in self.agents:
            self.observations[i] = self._get_obs(agent=i)

        if self.render_mode == "human":
            self.render()

        self.infos = {
            agent: {
                "action_mask": self._get_action_mask(agent)
            }
            for agent in self.agents
        }

        self._accumulate_rewards()

        #update agent selector
        self.agent_selection = self.state.current_player



    def observe(self, agent):
        return self._get_obs(agent)

    def _apply_action(self, action: int) -> GameState.GameState:
        """
        Decode integer action and call the appropriate engine function.

        DECLARE phase: actions 0-13
            # 1 declare 1 card, 0 of which are honest
            # 2 declare 1 card, 1 of which are honest
            # 3 declare 2 cards, 0 of which are honest
            # 4 declare 2 cards, 1 of which are honest

        CHALLENGE phase: actions 14, 15
            14 → challenge, 15 → pass
        """
        if action < 0:
            raise Exception(f"Invalid action: {action}")
        elif action < 14:
            decAction = declareActions[action] # qty, honest
            return Action.playCards(self.state, self._select_cards(decAction[0], decAction[1]))
        elif action == 14:
            return Action.challenge(self.state)
        elif action == 15:
            return Action.passChallenge(self.state)
        else:
            raise Exception(f"Unknown action: {action}")


    def relRank(self, card: GameState.Card) -> int:
        return (rankToInd[card.rank] - rankToInd[self.state.current_rank]) % 13


    def _select_cards(self, quantity: int, honest: int) -> tuple:
        """
        Given a (quantity, honest) decision, select concrete Card objects
        from the current player's hand with correct number of honest cards.

        Honest: pick `quantity` cards matching current_rank.
        Bluff:  pick `quantity` cards NOT matching current_rank.
                Heuristic: shed cards of ranks furthest from being required
                soon, given the cycling rank structure.

        Returns a tuple of Card objects suitable for Action.playCards().

        presume this is only called legitimately
        """
        # recall hand is sorted
        hand = self.state.players[self.state.current_player].hand
        # now just need to pivot around current rank
        i = 0
        while i < len(hand):
            if rankToInd[hand[i].rank] >= rankToInd[self.state.current_rank]:
                break
            i += 1
        # ex 1 2 2 3 5 7 with curr rank 4 gives i = 4
        # work backwards from it to bluff
        # go forwards from it to play honestly

        toPlay = []
        for j in range(honest):
            toPlay.append(hand[i + j])

        for j in range(quantity - honest):
            toPlay.append(hand[(i - j - 1) % len(hand)])

        return tuple(toPlay)

    # ------------------------------------------------------------------
    # Reward — to be implemented
    # ------------------------------------------------------------------
    def _retroactive_reward(self, acting_player: int, terminated: bool, action: int):
        # potentially offer rewards for bluff success? must be added retroactively
        # maybe not needed though... perhaps bad bluff will propagate through hand size penalty
        # to be explored
        raise NotImplementedError

    def _compute_reward(self, acting_player: int, terminated: bool, action: int):
        """
        Return scalar reward for acting_player.
        """

        w = 0.02
        a = w * np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, -self.state.prev_pile_size * self.state.last_truth, 0])
        #vectorized way to:
        # reward getting cards out
        # penalize a false challenge (picking up cards)
        # not do anything on pass
        # self.rewards[acting_player] += a[action]
        # if action == 14:
        #     self.rewards[acting_player] += 0.001 * (1 - 2 * self.state.last_truth)
            # penalized getting bluff called
        # for i in self.agents:
             # self.rewards[i] += -((self.state.turn_number // 6) ** 0.5) * 0.00001
        return


    def _get_obs(self, agent) -> np.ndarray:
        """
        Build the observation vector for the agent.
        See OBS_DIM layout at the top of this file.

        Call observe(self.state, self.state.current_player) to get
        a PlayerObservation, then vectorize it here.

        """
        # Observation vector layout:
        # [0:13]: counts of each rank, normalized by 4
        # [13:26]: cyclic encoding of current rank
        # [26]: pile size, normalized by deck size
        # [27]: claim quantity, normalized by 4 (0 if no claim)
        # [28]: current phase (0 for DECLARE, 1 for CHALLENGE)
        # Next NUM_PLAYERS: hand sizes, normalized by deck size

        vectorObs = np.zeros(OBS_DIM, np.float32)
        playerObs = GameState.observe(self.state, agent)
        # rank counts
        #TODO: optimize this. all the time is here
        for card in playerObs.player_hand:
            vectorObs[rankToInd[card.rank]] += 0.25

        # note the transformation to maintain [0, 1]
        for i in range(13):
            # (rankToInd[playerObs.current_rank] - i) will be zero on matching i
            vectorObs[13 + i] = 1 if rankToInd[playerObs.current_rank] == i else 0
        # goes through 25
        vectorObs[26] = rankToCyclicalSin[playerObs.current_rank]
        vectorObs[27] = rankToCyclicalCos[playerObs.current_rank]

        vectorObs[28] = playerObs.pile_size / 52 / NUM_DECKS

        vectorObs[29] = (playerObs.current_claim.quantity / 4) if playerObs.current_claim else 0

        vectorObs[30] = 0 if (playerObs.phase == GameState.Phase.DECLARE) else 1


        for i in range(NUM_PLAYERS):
            # curr, next, prev
            relIndex = (self.state.current_player + i) % 3
            vectorObs[31 + i] = playerObs.hand_sizes[relIndex] / 52 / NUM_DECKS

        return vectorObs


    def _get_action_mask(self, agent) -> np.ndarray:
        """
        Return a boolean array of shape (NUM_ACTIONS,) indicating which
        actions are valid in the current state.

        This mask is passed to MaskablePPO via the info dict. Invalid
        actions are set to -inf in the policy logits before sampling,
        so illegal actions are never taken.
        """
        if agent != self.state.current_player:
            return np.zeros(NUM_ACTIONS, np.int8)

        m = np.zeros(NUM_ACTIONS, np.int8)

        if self.state.current_phase == GameState.Phase.CHALLENGE:
            m[14] = 1
            m[15] = 1

        else: # if declare
            currHand = self.state.players[agent].hand
            # unmask  based on number of cards in hand
            matches = 0
            for card in currHand:
                if card.rank == self.state.current_rank:
                    matches += 1

            for i, (quantity, honest) in enumerate(declareActions):
                if quantity <= len(currHand) and honest <= matches:
                    m[i] = 1

        return m