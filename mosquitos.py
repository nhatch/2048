# A toy example of an agent using Q-learning in a Markov Decision Process.
# http://www.cs.indiana.edu/~gasser/Salsa/rl.html (the mosquitos example)
#
# Usage:
# a = Agent()
# a.choose()
#
# Observe how quickly the agent learns to go to state 4.

import random
import math

MDP = [
  [(0,-1), (1,0)],
  [(0,-1), (1,0), (2,0)],
  [(1,0), (2,0), (3,0)],
  [(2,0), (3,0), (4,1)],
  [(3,0), (4,1)]
]

gamma = 0.8
eta = 0.8
exploitation_rate = 0.1

class Agent:
  def __init__(self):
    self.state = 2
    self.age = 0
    self.Q = map(lambda (s): map(lambda(a): 0.0, s), MDP)

  def act(self, action):
    result = MDP[self.state][action]
    next_state = result[0]
    reward = result[1]
    new_Q = reward + gamma * max(self.Q[next_state])
    self.Q[self.state][action] = (1 - eta) * self.Q[self.state][action] + eta * new_Q
    self.state = next_state
    self.age += 1

  def choose(self):
    # http://scicomp.stackexchange.com/questions/1122/how-to-add-large-exponential-terms-reliably-without-overflow-errors
    exps = map(lambda(a): exploitation_rate * self.age * a, self.Q[self.state])
    max_exp = max(exps)
    vals = map(lambda(exp): math.exp(exp - max_exp), exps)
    denom = sum(vals)
    probs = map(lambda(val): val / denom, vals)
    print probs
    sample = random.random()
    for action in range(len(probs)):
      if sum(probs[:action+1]) >= sample:
        self.act(action)
        print "Now in state", self.state
        return
    # If we get here, `sample` hit a floating point rounding problem in `probs`.
    # This is unlikely. Just try again.
    choose(self)

# Seems like this doesn't sufficently prioritize exploration of new nodes. The goal is to collect reward as quickly as possible, which requires prioritizing learning how the board behaves. But how possible is it to learn the entire reward/state function?

