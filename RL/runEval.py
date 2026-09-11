from evaluate import *
import scenario_suite

ppo  = load_ppo_agent("./checkpointsPPO/policy_0100000.pt", device="cpu")
naive = make_naive_eval_agent("threshold")

# run_scenario(ppo, SCENARIOS["impossible_four_of_a_kind"])
# run_scenario(ppo, SCENARIOS["large_pile_high_claim"])
# run_scenario(ppo, SCENARIOS["agent_holds_all_four"])
# run_scenario(ppo, SCENARIOS["forced_bluff"])
# stats = quick_eval([lstm, oldlstm, ppo], n_games=500)
# print_all_stats(stats)
