"""
Source (open in private browser):
	https://towardsdatascience.com/how-to-code-the-value-iteration-algorithm-for-reinforcement-learning-8fb806e117d1
"""

from pprint import pprint
import numpy as np

'''
==================================================
Initial set up
==================================================
'''

# Hyperparameters
SMALL_ENOUGH = .01
GAMMA = 0.99
NOISE = 0.2
R = 100

# Define all states
all_states=[]
for i in range(3):
	for j in range(3):
			all_states.append((i,j))

# Define rewards for all states
rewards = {}
for i in all_states:
	if i == (0,0):
		rewards[i] = R
	if i == (0,2):
		rewards[i] = 10
	else:
		rewards[i] = -1

# Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
actions = {
	(0,0) : ('D', 'R'), 
	(0,1) : ('D', 'L', 'R'),    
	(0,2) : tuple(),
	(1,0) : ('U', 'D', 'R'),
	(1,1) : ('U', 'D', 'L', 'R'),
	(1,2) : ('U', 'D', 'L'),
	(2,0) : ('U', 'R'),
	(2,1) : ('U', 'L', 'R'),
	(2,2) : ('U', 'L')
}

# Define an initial policy
policy = {}
for s in actions.keys():
	policy[s] = np.random.choice(actions[s])

# Define initial value function 
V = {}
for s in all_states:
	if s in actions.keys():
		V[s] = 0
	if s == (0,0):
		V[s] = R
	if s == (0,2):
		rewards[i] = 10

print("-"*20)
print(f"all_states:")
pprint(all_states)
print("-"*20)
print(f"rewards:")
pprint(rewards)
print("-"*20)
print(f"actions:")
pprint(actions)
print("-"*20)
print(f"initial policy:")
pprint(policy)

'''
==================================================
Value Iteration
==================================================
'''

iteration = 0
smallest_biggest_change = None

while True:
	if iteration != 0 and iteration % 10000 == 0:
		print("-"*20)
		print(iteration)
		print(f"policy:")
		pprint(policy)
	biggest_change = 0
	for s in all_states:            
		if s in policy:
			
			old_v = V[s]
			new_v = 0
			
			for a in actions[s]:
				if a == 'U':
					nxt = [s[0]-1, s[1]]
				if a == 'D':
					nxt = [s[0]+1, s[1]]
				if a == 'L':
					nxt = [s[0], s[1]-1]
				if a == 'R':
					nxt = [s[0], s[1]+1]

				# Choose a new random action to do (transition probability)
				random_1 = np.random.choice([i for i in actions[s] if i != a])
				if random_1 == 'U':
					act = [s[0]-1, s[1]]
				if random_1 == 'D':
					act = [s[0]+1, s[1]]
				if random_1 == 'L':
					act = [s[0], s[1]-1]
				if random_1 == 'R':
					act = [s[0], s[1]+1]

				# Calculate the value
				nxt = tuple(nxt)
				act = tuple(act)
				v = rewards[s] + (GAMMA * ((1-NOISE)* V[nxt] + (NOISE * V[act]))) 
				if v > new_v: # Is this the best action so far? If so, keep it
					new_v = v
					policy[s] = a

	   # Save the best of all actions for the state                                
			V[s] = new_v
			biggest_change = max(biggest_change, np.abs(old_v - V[s]))

	# See if the loop should stop now  
	smallest_biggest_change = biggest_change if smallest_biggest_change is None else min([smallest_biggest_change, biggest_change])
	if biggest_change < SMALL_ENOUGH:
		break
	iteration += 1

print("-"*20)
print(f"end policy:")
pprint(policy)