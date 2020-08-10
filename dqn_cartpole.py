# General
import gym
import numpy as np
from collections import deque
import random

# Neural network 
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam

# Plotting
from matplotlib import pyplot as plt


# Hyperparameters
ENVNAME           = "CartPole-v1"
TRAINING_EPISODES = 1000
TEST_EPISODES     = 5
DISCOUNT_FACTOR   = 0.9999
EPSILON_INITIAL   = 1.0
EPSILON_DECAY     = 0.998
EPSILON_THRESHOLD = 0.01
MINIBATCH_SIZE    = 24
QUE_MAX_LEN       = 100000
LEARNING_RATE     = 0.0005
RUNS_TO_SOLVE     = 50
SOLVED_AVG_SCORE  = 350



class DQN():
	def __init__(self, env, model=None):
		self.memory           = deque(maxlen=QUE_MAX_LEN) # Holds: (s, a, r, s') for each step
		self.epsilon          = EPSILON_INITIAL
		self.actionSpace      = env.action_space.n
		self.observationSpace = env.observation_space.shape[0]
		# An alternative way of loading model is to use the loadModel() function
		self.model            = model   


	def act(self, state):
		""" Chooses whether to make an exploring move or a greedy move based on epsilon """
		if np.random.rand() < self.epsilon:
			# Non-greedy
			return random.randrange(self.actionSpace)
		# Greedy
		qValues = self.model.predict(state)
		return np.argmax(qValues[0])


	def greedyAct(self, state):
		""" Always chooses the action that maxamizes the q-value """
		qValues = self.model.predict(state)
		return np.argmax(qValues[0])


	def replay(self):
		""" 
		This is where the learning happens: 
			Agent uses saved experiences and adjusts itself based on 
			what the agent thought would happen vs what actually happened
		"""
		if len(self.memory) < MINIBATCH_SIZE:
			return
		# Take a random sample from the memory
		batch = random.sample(self.memory, MINIBATCH_SIZE)
		for state, action, reward, nextState, done in batch:
			newQ = reward
			if not done:
				# Bellman Equation:
				newQ = reward + DISCOUNT_FACTOR * np.amax(self.model.predict(nextState)[0])
			qValues = self.model.predict(state)          # Get predicted q-values
			qValues[0][action] = newQ                    # Update q-value for the action taken 
			self.model.fit(state, qValues, verbose=0)    # Make the model fit the new q-values
		
		self.epsilon *= EPSILON_DECAY
		self.epsilon  = max(EPSILON_THRESHOLD, self.epsilon)


	def loadModel(self, modelName):
		""" Load a saved model """
		self.model = load_model(modelName)
		print("Model Loaded. ")


	def saveModel(self, name):
		""" Call this function to save the current model """
		print('Saving Model...')
		self.model.save(name)


	def remember(self, state, action, reward, nextState, done):
		""" Saves game's information to agent's memory """
		self.memory.append((state, action, reward, nextState, done))


def buildModel(env, learningRate=LEARNING_RATE):
	# Create Model
	observationSpace = env.observation_space.shape[0]
	actionSpace = env.action_space.n
	
	model = Sequential()
	model.add(Dense(16, input_shape=(observationSpace,), activation="relu", init='he_uniform'))
	model.add(Dense(16, activation="relu", init='he_uniform'))
	model.add(Dense(16, activation="relu", init='he_uniform'))
	model.add(Dense(actionSpace, activation="linear"))
	model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
	model.summary()
	return model


def train(agent, env):
	""" Train the agent """
	observationSpace = env.observation_space.shape[0]
	episodeRewards = []
	episodeNum = 0
	meanLast10 = 0

	while True: 
		state = env.reset()
		state = np.reshape(state, [1, observationSpace])
		done = False
		step = 0
		episodeNum += 1
		while True:
			step += 1
			action = agent.act(state)

			nextState, reward, done, info = env.step(action)
			nextState = np.reshape(nextState, [1, observationSpace])

			# Give negative reward if it fails
			if not done:
				reward = reward
			else:
				reward = -reward
				agent.remember(state, action, reward, nextState, done)
				state = nextState
				break

			# Add the old state, action, rewards, new state, and done to the memory
			agent.remember(state, action, reward, nextState, done)
			state = nextState
			# Use experience replay to approximate q(s,a)
			agent.replay()
		
		episodeRewards.append(step)
		if len(episodeRewards) > 10:
			meanLast10 = np.mean(np.array(episodeRewards)[-10:])
		print("Episode Number: {}		Score: {}		Epsilon: {}		Memory: {}		Last 10: {}".format(
			   episodeNum, step, round(agent.epsilon, 3), len(agent.memory), meanLast10))
		if meanLast10 >= SOLVED_AVG_SCORE and len(episodeRewards) >= RUNS_TO_SOLVE:
			print('Environment solved! ')
			# Save model
			agent.saveModel('saved_models/model-meanScore-{}.h5'.format(meanLast10))
			# Save episode rewards to .npy file
			np.save('saved_models/rewards-meanScore-{}.npy'.format(meanLast10), np.array(episodeRewards))
			break


def test(agent, env):
	""" Visualize how well the agent plays the game """
	observationSpace = env.observation_space.shape[0]
	for _ in range(TEST_EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, observationSpace])
		done = False
		step = 0
		
		while not done:
			step += 1
			env.render()
			action = agent.greedyAct(state)
			# Take step
			nextState, reward, done, info = env.step(action)
			nextState = np.reshape(nextState, [1, observationSpace])
			# Update state
			state = nextState

		print("Total reward: ", step)


env     =    gym.make(ENVNAME)
model   =    buildModel(env)
agent   =    DQN(env, model=model)
#agent.loadModel('model-epsiode-SOLVED-episode-60.h5')
train(agent, env)
test(agent, env)

"""

env     =    gym.make(ENVNAME)
model   =    buildModel(env)
agent   =    DQN(env, model=model)
agent.loadModel('model-epsiode-749.h5')
agent.test()
"""

