import retro
import numpy as np
import cv2 
import neat
import pickle
from visualize import plot_stats, plot_species

# load the winner
winning_data = []
with open('winner/winner.pkl', 'rb') as input_file:
    while True:
        try:
            winning_data.append(pickle.load(input_file))
        except EOFError:
            break
winner, config, stats = winning_data
# ---

p = neat.Population(config)
winner_net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)

# re-establish environment
env = retro.make('Frogger-Genesis', '1Player.Level1')
ob = env.reset()
ac = env.action_space.sample()
inx, iny, inc = env.observation_space.shape
inx = int(inx/8)
iny = int(iny/8)
current_max_fitness = 0
fitness_current = 0
frames = 0
imgarray = []
done = False

while not done:
	env.render() # render the ROM to see on your screen
	ob = cv2.resize(ob, (inx, iny)) # resize the observation space
	ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # convert to greyscale
	ob = np.reshape(ob, (inx, iny)) # reshape the observation space
	imgarray = np.ndarray.flatten(ob) # flatten the observation space
	
	
	# generate the output from the neural network
	nnOutput = winner_net.activate(imgarray)
	fill = [0] * 4 # used for frogger
	action = fill + nnOutput + fill # fill in the other space
 
	# step the environment forward and record the observation, reward, done state, and info
	ob, rew, done, info = env.step(action)
	
	# record the fitness, generic to all games
	fitness_current += rew
	
	# if the fitness is greater than the current max fitness, reset the counter (the frame counter)
	if fitness_current > current_max_fitness:
		current_max_fitness = fitness_current
		frames = 0
	else:
		frames += 1
		
	if done or frames == 250:
		# if the episode is done or 250 frames are recorded, end the episode
		done = True
		print(fitness_current)
	
	winner.fitness = fitness_current

# draw_net(config, winner, True)
# draw_net(config, winner, True, prune_unused=True)
plot_stats(stats, ylog=False, view=True)
plot_species(stats, view=True)