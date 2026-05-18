# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        # This performs self.iterations rounds of value iteration.
        for i in range(self.iterations):
            # Create a new Counter to store the values for the next iteration (V_k+1).
            # This is the "batch" version of value iteration, where we calculate
            # all new values based on the full set of old values (V_k)
            new_values = util.Counter()
            # Iterate over all states in the MDP.
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                possible_actions = self.mdp.getPossibleActions(state)
                # If there are no actions, the value of the state is 0.
                if not possible_actions:
                    continue

                # Calculate the Q-value for each possible action from the current state.
                q_values = [self.computeQValueFromValues(state, action) for action in possible_actions]
                # The new value of the state is the maximum Q-value among all possible actions.
                new_values[state] = max(q_values)
            
            # After computing the new values for all states, update self.values.
            # This completes one iteration of the "batch" update.
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Initialize the Q-value for the (state, action) pair to 0.
        q_value = 0
        # Get all possible transition states and their probabilities for the given state and action.
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        
        # For each possible outcome (nextState, prob), calculate its contribution to the Q-value.
        for nextState, prob in transitions:
            # Get the reward for transitioning to the next state.
            reward = self.mdp.getReward(state, action, nextState)
            # The Q-value is the sum over all possible next states of:
            # probability * (immediate reward + discounted value of the next state)
            # This is the formula: Q(s, a) = Σ_s' T(s, a, s') * [R(s, a, s') + γ * V(s')]
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # A terminal state has no actions, so the policy should return None.
        if self.mdp.isTerminal(state):
            return None

        # Get all possible actions from the current state.
        possible_actions = self.mdp.getPossibleActions(state)
        # If there are no actions, there is no policy.
        if not possible_actions:
            return None

        # Use a Counter to store the Q-value for each action.
        q_values = util.Counter()
        # Calculate the Q-value for each possible action.
        for action in possible_actions:
            q_values[action] = self.computeQValueFromValues(state, action)
        
        # The best action is the one with the highest Q-value.
        # The argMax() method of the Counter class is perfect for this.
        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Performs asynchronous cyclic value iteration. In each iteration, it
        updates the value of a single state, cycling through all states.
        """
        # Get all states from the MDP.
        states = self.mdp.getStates()
        num_states = len(states)

        # Loop for the specified number of iterations.
        for i in range(self.iterations):
            # Determine the state to update for this iteration in a cyclic order.
            state = states[i % num_states]

            # Skip terminal states as their values are fixed at 0.
            if self.mdp.isTerminal(state):
                continue

            # Find the action that maximizes the Q-value.
            actions = self.mdp.getPossibleActions(state)
            if not actions:
                continue  # No actions possible, value is 0.

            # Update the state's value to the maximum Q-value from any possible action.
            self.values[state] = max([self.computeQValueFromValues(state, action) for action in actions])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # Get all states from the MDP.
        # Initilize the predecessors of each state to an empty set. 
        # A predecessor of a state s is any state that can transition to s with non-zero probability.
        predecessors = {state: set() for state in states} 
        # Initialize priority queue
        pq = util.PriorityQueue()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
