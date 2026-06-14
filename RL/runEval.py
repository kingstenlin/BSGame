from evaluate import quick_eval, load_ppo_agent, load_lstm_agent, make_naive_eval_agent, print_all_stats

ppo  = load_ppo_agent("./ckpts/checkpointsWLPoolRelHands/policy_0100000.pt", device="cpu")
ppo2 = load_ppo_agent("./ckpts/checkpointsWLPoolRelHands/policy_0060000.pt", device="cpu")
lstm = load_lstm_agent("./checkpoints_lstmDetach/policy_0100000.pt", device="cpu")
oldlstm = load_lstm_agent("./checkpoints_lstm1/policy_0065024.pt", device="cpu")
stats = quick_eval([lstm, oldlstm, ppo], n_games=500)
print_all_stats(stats)
