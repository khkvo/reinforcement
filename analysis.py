# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0 # Change the noise to 0.0 so that the agent will go straight to the exit and not risk the cliff
    return answerDiscount, answerNoise

def question3a():
    # Prefer the close exit (+1), risking the cliff (-10).
    # A low discount factor makes the agent "short-sighted," prioritizing the closer +1 reward.
    # A large negative living reward penalizes every step, forcing the agent to take the
    # shortest path along the cliff.
    answerDiscount = 0.1
    answerNoise = 0.2
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10).
    # A low discount factor again makes the agent prefer the closer +1 reward.
    # To avoid the cliff, we remove the penalty for taking time by setting the living reward to 0.
    # The default noise makes the cliff path risky, so the agent chooses the longer, safer path.
    answerDiscount = 0.1
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # Prefer the distant exit (+10), risking the cliff (-10).
    # A high discount factor makes the agent "far-sighted," valuing the distant +10 reward more.
    # To make it risk the cliff, we eliminate the risk by setting noise to 0.
    # A small negative living reward then encourages it to take the shorter of the two now-safe paths.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Prefer the distant exit (+10), avoiding the cliff (-10).
    # A high discount factor makes the agent value the distant +10 reward.
    # With the default noise, the agent is risk-averse. A zero living reward means it doesn't
    # mind taking the longer, safer path to avoid the cliff.
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # Avoid both exits and the cliff (so an episode should never terminate).
    # A positive living reward incentivizes the agent to stay in the game.
    # A discount of 1.0 means future rewards don't lose value, so an infinite stream of
    # positive rewards has infinite value, which is better than any finite terminal reward.
    # Noise is set to 0 to ensure it can execute a safe loop without accidentally terminating.
    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
