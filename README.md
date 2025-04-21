# Portfolio Optimization with Reinforcement Learning

This project uses **Reinforcement Learning** (PPO) to optimize stock portfolio allocation and also combines it with **sentiment analysis** from SEC filings.

---

## Sentiment Analysis

We use **FinBERT** to get a sentiment from the “Notes to Financial Statements” section in SEC 10-K and 10-Q filings.

This sentiment is used to help make portfolio decisions by showing any possible risks, optimism, or uncertainty from the companies themselves.

Outputs:

- `per_note.csv` — one row per disclosure block
- `per_filing.csv` — one row per filing (with average sentiment)

---

## Makefile Commands

Run these commands from the project root:

- `make`: Main command: download + run sentiment pipeline
- `make get-notes`: Download latest SEC notes dataset
- `make run-pipeline`: Run sentiment pipeline on most recent dataset

---

## Requirements

See `env.yaml` to set up the conda environment:

```bash
conda env create -f env.yaml
conda activate RL
```

---
