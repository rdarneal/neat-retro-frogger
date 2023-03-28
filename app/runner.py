import retro        # https://retro.readthedocs.io/en/latest
import numpy as np  # https://numpy.org/doc/stable/
import cv2          # https://docs.opencv.org/
import neat         # https://neat-python.readthedocs.io/en/latest/

# initiate the gym environment
# browse available defaults @ https://github.com/openai/retro/tree/master/retro/data/stable
# Get ROMs @ https://archive.org/details/No-Intro-Collection_2016-01-03_Fixed
# Preload the ROMs from python shell using `python -m retro.import /path/to/your/ROMs/directory/`
env = retro.make('Frogger-Genesis', state='1Player.Level1')

# 1d array to record the observed environment
imgarray = []

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        # reset the gym environment
        ob = env.reset()
        
        # randomly select an action
        ac = env.action_space.sample()
        
        # get the observation space (x, y, color)
        inx, iny, inc = env.observation_space.shape

		# scale the input to 28 x 40 = 1120 (.config[num_inputs]])
        inx = int(inx/8) #224 -> 28
        iny = int(iny/8) #320 -> 40
        
        # create the recurrent neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        # set parameters for fitness function
        current_max_fitness = 0
        fitness_current = 0
        frames = 0        
        done = False
        
        # # Uncomment to display each frame how the NN sees it
        # cv2.namedWindow('main', cv2.WINDOW_NORMAL)

        while not done:
            env.render() # render the ROM to see on your screen
            ob = cv2.resize(ob, (inx, iny)) # resize the observation space
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # convert to greyscale
            ob = np.reshape(ob, (inx, iny)) # reshape the observation space
            imgarray = np.ndarray.flatten(ob) # flatten the observation space
            
            # # Use this to see what the NN is seeing   
            # scaledimg = cv2.cvtColor(ob, cv2.COLOR_GRAY2BGR)
            # scaledimg = cv2.resize(scaledimg, (iny, inx))
            # cv2.imshow('main', scaledimg)
            # cv2.waitKey(1)
            
            # generate the output from the neural network
            nnOutput = net.activate(imgarray)
            fill = [0] * 4
            action = fill + nnOutput + fill
            # step the environment forward and record the observation, reward, done state, and info
            ob, rew, done, info = env.step(action)
            
            # # Specific to Sonic 2 ------------------
            # # get the x position of the character from the info
            # xpos = info['x']
            # xpos_end = info['screen_x_end']
            
            # # if the character moves forward, increase the fitness and record the max x position
            # if xpos > xpos_max:
            #     fitness_current += 1
            #     xpos_max = xpos
            
            # # if the character has reached the end of the level, increase the fitness and end the episode
            # if xpos == xpos_end and xpos > 500:
            #     fitness_current += 100000
            #     done = True
            # # --------------------------------------
            
            # record the fitness, generic to all games
            fitness_current += rew
            
            # if the fitness is greater than the current max fitness, reset the counter (the frame counter)
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                frames = 0
            else:
                frames += 1
                
            if info['lives'] == 0 or frames > 250:
                # if the episode is done or 250 frames are recorded, end the episode
                done = True
                print(genome_id, fitness_current)
            
            genome.fitness = fitness_current
                
# load the config file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     '.config')
# create the population
p = neat.Population(config)

# add reporters to show progress in the terminal
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

# 
winner = p.run(eval_genomes)