# Project 3: Reinforcement Learning

UC Berkeley CS 188 Pacman AI project focused on reinforcement learning. Students implement value iteration and Q-learning agents, then apply them to Gridworld, a simulated robot crawler, and Pacman.

## Student Files to Edit

- `valueIterationAgents.py` - Value iteration agent for Gridworld
- `qlearningAgents.py` - Q-learning agents for Gridworld, Crawler, and Pacman
- `analysis.py` - Answers to analysis questions (parameter tuning)

## Running

### Gridworld (manual control)

```bash
python3 gridworld.py -m
```

### Autograder

```bash
python3 autograder.py
```

Grade a specific question:

```bash
python3 autograder.py -q q1
```

### Crawler

```bash
python3 crawler.py
```

### Pacman with Q-learning

```bash
python3 pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

## Key Concepts

- **Value Iteration** (offline planning): computes optimal policy from a known MDP
- **Q-Learning** (online learning): learns action values from experience without a model
- **Epsilon-Greedy**: exploration strategy balancing random exploration with learned policy
- **Approximate Q-Learning**: uses feature extractors to generalize across states

## Project Structure

| File | Description |
|------|-------------|
| `mdp.py` | MDP interface definition |
| `environment.py` | Environment interface |
| `learningAgents.py` | Base classes for learning agents |
| `featureExtractors.py` | Feature extractors for approximate Q-learning |
| `gridworld.py` | Gridworld MDP implementation and main driver |
| `crawler.py` | Crawler robot simulation |
| `pacman.py` | Pacman game engine |
| `autograder.py` | Project autograder |
| `test_cases/` | Test cases for each question |

## Attribution

The Pacman AI projects were developed at UC Berkeley by John DeNero and Dan Klein. Student-side autograding by Brad Miller, Nick Hay, and Pieter Abbeel. More info at http://ai.berkeley.edu.
