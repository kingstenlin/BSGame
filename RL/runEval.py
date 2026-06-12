from evaluate import quick_eval, load_ppo_agent, load_lstm_agent, make_naive_eval_agent, print_all_stats

ppo  = load_ppo_agent("./ckpts/checkpointsWLPoolRelHands/policy_0100000.pt", device="cpu")
lstm = load_lstm_agent("./checkpoints_lstm/policy_0060000.pt", device="cpu")
stats = quick_eval([lstm, lstm, ppo], n_games=500)
print_all_stats(stats)

# ┌──────────────────────────────────────────────────────┐
#   │  Matchup: PPO(ep=100000) vs LSTM(ep=60000) vs Threshold│
#   ├──────────────────────────────────────────────────────┤
#   │  Games:      500     Completed: 500     Truncated: 0    │
#   │  Moves/game: mean 811.0   median 567.5             │
#   ├──────────────────────────────────────────────────────┤
#   │  Agent                 Wins   Win rate   Seat  │
#   ├──────────────────────────────────────────────────────┤
#   │  PPO(ep=100000)         119     23.8%     p0  │
#   │  LSTM(ep=60000)         381     76.2%     p1  │
#   │  Threshold                0      0.0%     p2  │
#   └──────────────────────────────────────────────────────┘