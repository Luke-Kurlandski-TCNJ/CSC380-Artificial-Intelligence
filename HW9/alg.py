"""
Mathematical foundations: 
	http://incompleteideas.net/book/ebook/node44.html
"""

from pprint import pprint
import sys
from typing import Dict, List, Tuple

########################################################################
# Hyperparameters
########################################################################

SMALL = .001		# small value to test for convergence
DISCOUNT = 0.99		# discount factor (frequently referred to as gamma)
NOISE = 0.2			# chance of making a random move
R = 100				# r value for the top left square

########################################################################
# Set up states, rewards, actions, policy, and the value function
########################################################################

# Define all states
states = [
	(0,0), (0,1), (0,2),
	(1,0), (1,1), (1,2),
	(2,0), (2,1), (2,2),
]

# Define rewards for all states
rewards = {
	(0,0) : R,  (0,1) : -1, (0,2) : 10,
	(1,0) : -1, (1,1) : -1, (1,2) : -1,
	(2,0) : -1, (2,1) : -1, (2,2) : -1,
}

value_function = {
	(0,0) : 0, (0,1) : 0, (0,2) : 0,
	(1,0) : 0, (1,1) : 0, (1,2) : 0,
	(2,0) : 0, (2,1) : 0, (2,2) : 0,
}

# Dictionnary of possible actions, (0,2) is a terminal state (no action)
# This is viewed from the perspective of 0,0 being the top left corner
actions = {
	(0,0) : ('D', 'R'), 
	(0,1) : ('D', 'L', 'R'),    
	(1,0) : ('U', 'D', 'R'),
	(1,1) : ('U', 'D', 'L', 'R'),
	(1,2) : ('U', 'D', 'L'),
	(2,0) : ('U', 'R'),
	(2,1) : ('U', 'L', 'R'),
	(2,2) : ('U', 'L')
}

# Define an initial policy
policy = {
	(0,0) : 'D',
	(0,1) : 'D',  
	(1,0) : 'D',
	(1,1) : 'D',
	(1,2) : 'D',
	(2,0) : 'D',
	(2,1) : 'D',
	(2,2) : 'D',
}

########################################################################
# Useful functions
########################################################################

def print_policy(p:Dict[Tuple[str, str], str]):
	"""Print the policy in a grid-like format.

	Args:
		p (Dict[Tuple[str, str], str]): the policy to print
	"""
	p[(0,2)] = 'X'
	arr = [
		[p[(0,0)], p[(0,1)], p[(0,2)]],
		[p[(1,0)], p[(1,1)], p[(1,2)]],
		[p[(2,0)], p[(2,1)], p[(2,2)]],
	]
	for a in arr:
		print(a)

def possible_next_states(
	s:Tuple[str, str], a:str
) -> List[Tuple[Tuple[str, str], float]]:
	"""Return the possible next states from a given state and the 
		corresponding probability of each state given an action.

	Args:
		s (Tuple[str, str]): current state
		a (str): action to takem or attempt

	Returns:
		List[Tuple[str, str], float]: a list of possible next states and
			the corresponding probability of that state
	"""
	if s not in actions:
		return []

	possible_next_states = []

	# Add the attempted state and the prob of making the desired move
	if a == 'U':
		possible_next_states.append(((s[0]-1, s[1]), 1-NOISE))
	elif a == 'D':
		possible_next_states.append(((s[0]+1, s[1]), 1-NOISE))
	elif a == 'L':
		possible_next_states.append(((s[0], s[1]-1), 1-NOISE))
	elif a == 'R':
		possible_next_states.append(((s[0], s[1]+1), 1-NOISE))

	# Add the other possible states and the prob of undesired move
	undesired_prob = NOISE / (len(actions[s]) - 1)
	for option in actions[s]:
		if option != a:
			if option == 'U':
				possible_next_states.append(((s[0]-1, s[1]), undesired_prob))
			elif option == 'D':
				possible_next_states.append(((s[0]+1, s[1]), undesired_prob))
			elif option == 'L':
				possible_next_states.append(((s[0], s[1]-1), undesired_prob))
			elif option == 'R':
				possible_next_states.append(((s[0], s[1]+1), undesired_prob))

	return possible_next_states

########################################################################
# Value Iteration Algorithm
########################################################################

# Do until stopping condition occurs
delta = sys.maxsize
while delta > SMALL:
	delta = 0

	# Loop through every state that has an action, ie, is not terminal
	for s in states:
		if s not in actions: 
			continue

		v = value_function[s]					# current value function
		max_over_actions = -1 * sys.maxsize		# storage for loop

		# Loop through every possible action for the current state
		for a in actions[s]:
			# Compute the sum over the next possible states given their
				# corresponding probabilities of occuring
			sum_over_next_states = 0
			for s_prime, prob_of_s_prime in possible_next_states(s, a):
				sum_over_next_states = sum_over_next_states \
					+ prob_of_s_prime * (rewards[s_prime] \
					+ DISCOUNT * value_function[s_prime])
			# If this is the optimal action encountered thus far, update
				# the policy and the tracking variable
			if sum_over_next_states > max_over_actions:
				policy[s] = a
				max_over_actions = sum_over_next_states
		
		# Update the value function and the stopping value
		value_function[s] = max_over_actions
		delta = max(delta, abs(v - value_function[s]))

# Print the value function and the policy
print("-"*20)
pprint(value_function)
print("-"*20)
print_policy(policy)