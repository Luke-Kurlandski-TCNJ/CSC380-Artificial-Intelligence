"""
Homework 9: Value Iteration Algorithm for Markov Decision Process
"""

from copy import deepcopy
from pprint import pprint
import sys
from typing import Dict, List, Tuple

########################################################################
# Hyperparameters
########################################################################

SMALL = .001		# small value to test for convergence
DISCOUNT = 0.99		# discount factor (frequently referred to as gamma)
NOISE = 0.2			# chance of making a random move

########################################################################
# Useful function
########################################################################

def print_policy(p:Dict[Tuple[int, int], str]):
	"""Print the policy in a grid-like format.

	Args:
		p (Dict[Tuple[int, int], str]): the policy to print
	"""
	p[(0,2)] = 'X'
	arr = [
		[p[(0,0)], p[(0,1)], p[(0,2)]],
		[p[(1,0)], p[(1,1)], p[(1,2)]],
		[p[(2,0)], p[(2,1)], p[(2,2)]],
	]
	for a in arr:
		print(a)

########################################################################
# Value Iteration Algorithm
########################################################################

def possible_next_states(
	s:Tuple[int, int], 
	a:str, 
	actions:Dict[Tuple[int, int], Tuple[str,]]
) -> List[Tuple[Tuple[int, int], float]]:
	"""Return the possible next states from a given state and the 
		corresponding probability of each state given an action.

	Args:
		s (Tuple[int, int]): current state
		a (str): action to takem or attempt
		action (Dict[Tuple[int, int], Tuple[str,]]): all actions for
			the MDP

	Returns:
		List[Tuple[int, int], float]: a list of possible next states and
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
	undesired = NOISE / (len(actions[s]) - 1)
	for option in actions[s]:
		if option != a:
			if option == 'U':
				possible_next_states.append(((s[0]-1, s[1]), undesired))
			elif option == 'D':
				possible_next_states.append(((s[0]+1, s[1]), undesired))
			elif option == 'L':
				possible_next_states.append(((s[0], s[1]-1), undesired))
			elif option == 'R':
				possible_next_states.append(((s[0], s[1]+1), undesired))

	return possible_next_states

def value_iteration(
	states:List[Tuple[int, int]], 
	actions:Dict[Tuple[int, int], Tuple[str,]], 
	rewards:Dict[Tuple[int, int], int], 
	policy:Dict[Tuple[int, int], str], 
	value_function:Dict[Tuple[int, int], int]
) -> Dict[Tuple[int, int], int]:
	"""Perform the value iteration algorithm to find the optimal policy.

	Args:
		states (List[Tuple[int, int]]): states for MPD
		actions (Dict[Tuple[int, int], Tuple[str,]]): actions for MPD
		rewards (Dict[Tuple[int, int], int]): rewards for MPD
		policy (Dict[Tuple[int, int], str]): intitial policy for MPD
		value_function (Dict[Tuple[int, int], int]): initial value 
			function for MPD

	Returns:
		Dict[Tuple[int, int], str]: final policy for MPD
	"""

	policy = deepcopy(policy)

	# Do until stopping condition occurs
	delta = sys.maxsize
	while delta > SMALL:
		delta = 0

		# Loop through every state that has an action, ie, not terminal
		for s in states:
			if s not in actions: 
				continue

			v = value_function[s]				# current value function
			max_over_actions = -1 * sys.maxsize	# storage for loop

			# Loop through every possible action for the current state
			for a in actions[s]:
				# Compute the sum over the next possible states given
					# corresponding probabilities of occuring
				sum_over_next_states = 0
				for s_, p in possible_next_states(s, a, actions):
					sum_over_next_states += p * (rewards[s_] \
						+ DISCOUNT * value_function[s_])
				# If this is the optimal action encountered thus far,
					# update the policy and the tracking variable
				if sum_over_next_states > max_over_actions:
					policy[s] = a
					max_over_actions = sum_over_next_states
			
			# Update the value function and the stopping value
			value_function[s] = max_over_actions
			delta = max(delta, abs(v - value_function[s]))

	return policy

########################################################################
# Set up the Markov Decision Process and run the algorithm
########################################################################

def main(r:int):

	# Define all states
	states = [
		(0,0), (0,1), (0,2),
		(1,0), (1,1), (1,2),
		(2,0), (2,1), (2,2),
	]

	# Define rewards for all states
	rewards = {
		(0,0) : r,  (0,1) : -1, (0,2) : 10,
		(1,0) : -1, (1,1) : -1, (1,2) : -1,
		(2,0) : -1, (2,1) : -1, (2,2) : -1,
	}

	value_function = {
		(0,0) : 0, (0,1) : 0, (0,2) : 0,
		(1,0) : 0, (1,1) : 0, (1,2) : 0,
		(2,0) : 0, (2,1) : 0, (2,2) : 0,
	}

	# Dictionnary of possible actions, (0,2) is a terminal state
	# This is viewed from the perspective of 0,0 being the top left
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

	policy = value_iteration(
		states, 
		actions, 
		rewards, 
		policy, 
		value_function
	)

	# Print the value function and the policy
	print("-"*20)
	print(r)
	print_policy(policy)

if __name__ == "__main__":
	main(100)
	main(-3)
	main(0)
	main(3)
