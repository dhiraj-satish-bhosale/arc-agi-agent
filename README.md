# ARC-AGI-3 Agent Experiments

My experiments with LLM-based agents on the [ARC-AGI-3](https://arcprize.org/) benchmark: interactive reasoning games where an agent has to discover the rules by playing.

## What is mine vs borrowed

The benchmark harness and starter templates are not my code:

- `arc-prize-2026/` - the official [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) starter kit, vendored for reference.
- `ARC3-solution-main/` - an open-source ARC-AGI-3 solution repo, vendored for reference.

My own work lives at the repo root:

- `custom_agent.py` / `custom_agents/` - my agent built on the starter templates, experimenting with enhanced reasoning prompts.
- `utils.py` - helpers around the agent loop.
- `Makefile` - `make install`, `make enhanced`, `make tensorboard` shortcuts for running agents.

## Run it

```bash
make install      # create the venv and install dependencies
make enhanced     # run my enhanced agent
```

## Status

Exploratory. This is a sandbox for learning how LLM agents behave on interactive reasoning tasks, not a finished entry.
