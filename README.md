# QuantMind: Autonomous AI Quant Trading Intelligence

Developed by Sherif Khaled.

> ⚠️ **Paper trading only by default.** No system guarantees profit. Every figure the app shows comes from simulated trading or historical tests. Live trading is locked in code and can't be switched on from the app.

## 🔔 The trading assistant (how you use it)

The brain studies the market every hour. When it finds an opportunity that passed every test, you get a **Telegram notification**. It shows up over whatever app you have open (Binance or any other):

```
🟢 Buy signal: BTC
💵 Price now: 97,250
✅ Buy only if the price is below: 97,700
📦 Position size: 4% of your capital
🛑 Stop-loss: 94,800  (place it as a Stop-Loss order in the app right away)
🎯 First target: 100,900  •  Second target: 104,550
⏳ The signal is valid until 16:00
```

After that it tells you when to sell: 🔴 *"Sell now"*, 🎯 *"First target reached, move your stop to break-even"*, or ⚠️ *"Price hit the stop-loss"*.
The app's **Signals** tab shows the open signal, buttons to say whether you entered the trade, and an honest **record** of the result of every signal.
The default is **Spot**: plain buying and selling with no leverage and no short selling. To allow short signals, set `QM_MARKET_TYPE=futures`. That's not recommended for beginners.

## How it works (in plain terms)

| Part | What it does |
|---|---|
| 🧠 **The brain** (`backend/`) | Studies the market every hour, invents strategies, tests them, and runs a debate between agents before any decision |
| ⚙️ **GitHub Actions** | Runs the brain hourly for free and saves its memory on the `state` branch |
| 📱 **Android app** (`android-app/`) | Shows the paper portfolio, what the brain is thinking, the lab, what it learned, and an emergency stop button |
| 💬 **Telegram** (optional) | Sends a summary when something happens |

## Setup (about 10 minutes)

1. Create a new repository on GitHub and upload this folder to it.
2. Go to **Settings → Actions → General** and choose *Read and write permissions*.
3. (Optional) Go to **Settings → Secrets → Actions** and add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`. You can also add `ANTHROPIC_API_KEY` so the explanations read more naturally; the brain only uses it to explain decisions, never to make them.
4. Go to **Actions → QuantMind cycle → Run workflow** to run the first cycle. After that it runs on its own every hour.
5. **Actions → Build Android app → Run workflow**. When it finishes, open **Releases → android-latest** on your phone and download `QuantMind.apk`.
6. In the app's **Settings** screen, enter `owner/repo` and a fine-grained token with *Contents: read* and *Actions: read & write* on this repository only.

> 💡 If the repository is private, the free Actions minutes (2000 per month) can run out because the brain runs every hour. A public repository gets unlimited minutes, and no secrets live in it because the keys are stored in Secrets. GitHub also pauses scheduled workflows on a repository with no activity for 60 days.

## Architecture

```
backend/quantmind/
  data/          provider interface + Binance (data only) + CSV + synthetic (tests only)
  analysis/      own indicator library + causal feature engine (feature space = lab vocabulary)
  regime/        interpretable regime labels + GMM clustering + IsolationForest novelty
  strategies/    rule language v2 (feature-vs-feature rules, regime genes, trailing stops,
                 multi-timeframe 4h/1d features) on a numba-compiled engine (~0.7 ms per
                 2 years of 1h bars) + walk-forward ML strategies (gradient boosting, purged)
  backtest/      next-bar execution, fees + vol-scaled slippage + market impact + funding
  validation/    3-way split w/ embargo, walk-forward, Deflated Sharpe, permutation test,
                 causality (look-ahead) check, Monte-Carlo drawdown
  lab/           genetic search (niches, immigrants) + GP Bayesian optimisation + ML configs
                 -> 8 gates: validation, causality, stability (>=4/6 periods), robustness
                 (neighbouring params), 2x-cost stress, holdout vs buy & hold, deflated
                 Sharpe, permutation/timing test. Registry keeps every rejection + reason
  agents/        9 specialists + contrarian + performance auditor + 3-round debate
  portfolio/     ensemble of validated strategies: regime fit x shrunk Sharpe / vol,
                 de-duplicated by correlation
  signals/       trading assistant: buy / sell / stop / target instructions + track record
  risk/          independent risk manager with veto, kill switch, emergency shutdown
  execution/     paper broker (persisted) + live gateway (locked)
  forecast/      regime-conditional scenarios + probability shown ONLY if calibrated OOS
  research/      hypothesis generation from measurements, lead-lag, correlation breaks,
                 prioritized backlog, strategy decay monitor
  orchestrator/  the OBSERVE → … → IMPROVE cycle, Arabic narrator, snapshot for the app
  api/           optional FastAPI server mode (+ WebSocket event stream)
```

### Spec coverage, and what differs from it on purpose

- **The debate isn't a vote.** Weights come from each agent's measured track record (a Beta prior on its out-of-sample hit rate) multiplied by its confidence after cross-examination and the contrarian's challenges. Strong disagreement means no trade.
- **Only validated strategies can open trades.** The discretionary agents (technical, price action, and so on) are themselves untested, so they can strengthen, weaken or block a trade but never start one. This applies the spec's rule "Never deploy unvalidated strategies" to the agents too.
- **Selection bias is counted honestly.** Every strategy variant tested is recorded, and the Deflated Sharpe gate is corrected for the cumulative number of holdout tests. That count is never reset, so reusing the holdout keeps getting harder.
- **Proof that the gates work** (`tests/test_lab.py`): on a random walk the lab validates nothing, and on data with a planted real edge it finds it.
- **Probabilities only when calibrated.** If the model doesn't beat the base rate out-of-sample, the output says "uncalibrated" instead of showing a number.
- **Agents abstain when they have no data** (sentiment, fundamentals, order book) rather than making it up.
- **Storage is JSON on a git branch instead of PostgreSQL/Redis.** It's free, every change is versioned, and it suits GitHub Actions. The `Store` interface is small, so a Postgres backend can be added later. Redis is supported for the event bus if `QM_REDIS_URL` is set.
- **The dashboard is the Android app** (Capacitor) rather than Next.js, at the owner's request.

### Data note
Market data comes from `data-api.binance.vision`, which is reachable from GitHub's US runners. Futures data (funding, open interest) on `fapi.binance.com` is blocked for US IPs, so on GitHub Actions the derivatives agent abstains. Server mode outside the US gets the full data.

## Commands

```bash
cd backend && pip install -r requirements.txt
python -m pytest -q tests                      # 22 tests: causality (incl. ML + multi-timeframe), costs,
                                               # intrabar stops, lab honesty, ensemble, advisor, risk, debate
python -m quantmind.cli cycle --demo           # offline run on synthetic data (labelled as such)
python -m quantmind.cli cycle                  # real Binance data
python -m quantmind.cli control --pause "why"  # emergency stop;  --resume ; --level 1|2|3
python -m quantmind.cli control --taken BTCUSDT   # mark that you entered the trade (the app does this for you)
docker compose up -d                           # optional server mode on :8000
```

## Live trading (level 4)

It stays locked until all of these are true:
1. Several months of paper-trading results that you've reviewed.
2. `QM_LIVE_TRADING_ENABLED=true` and an authorization token set by hand on the server. It can't be enabled from the app or the API.
3. A hard capital ceiling, `QM_LIVE_MAX_CAPITAL_USD`.
4. Approval from the risk manager on every order.

Even with all of that in place, the code that sends orders in `execution/live.py` is deliberately left as a stub. It needs to be completed and reviewed separately.
