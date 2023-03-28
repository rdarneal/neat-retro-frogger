import retro        # https://retro.readthedocs.io/en/latest
import numpy as np  # https://numpy.org/doc/stable/
import cv2          # https://docs.opencv.org/
import neat         # https://neat-python.readthedocs.io/en/latest/
import pickle       # https://docs.python.org/3/library/pickle.html


class Worker(object):
    def __init__(self, genome, config, game='Frogger-Genesis', state='1Player.Level1'):
        self.genome = genome
        self.config = config
        self.game = game
        self.state = state
        #TODO: add exception if game or state is not found
    
    def work(self):
        # initiate the gym environment
        # browse available defaults @ https://github.com/openai/retro/tree/master/retro/data/stable
        # Get ROMs @ https://archive.org/details/No-Intro-Collection_2016-01-03_Fixed
        # Preload the ROMs from python shell using `python -m retro.import /path/to/your/ROMs/directory/`
        self.env = retro.make(self.game, self.state)
        self.env.reset()
        # get the observatino space
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        # get x, y of the observation space
        inx, iny, _ = ob.shape

		# scale the input to 28 x 40 = 1120 (.config[num_inputs]])
        inx = int(inx/8) #224 -> 28
        iny = int(iny/8) #320 -> 40
        #TODO: confirm the scale matches the config file, e.g. ROM screen size
        
        # create the recurrent neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        
        # set parameters for fitness function
        current_max_fitness = 0
        fitness_current = 0
        frames = 0
        imgarray = []
        done = False
        
        # # Uncomment to display each frame how the NN sees it
        # cv2.namedWindow('main', cv2.WINDOW_NORMAL)

        while not done:
            # self.env.render() # render the ROM to see on your screen
            ob = cv2.resize(ob, (inx, iny)) # resize the observation space
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # convert to greyscale
            ob = np.reshape(ob, (inx, iny)) # reshape the observation space
            imgarray = np.ndarray.flatten(ob) # flatten the observation space
            
            nnOutput = net.activate(imgarray) # generate the output from the neural network
            fill = [0] * 4 # used for frogger since we only need up, down, left, right
            action = fill + nnOutput + fill # fill the list with 0s to match the action space
            
            # step the environment forward and record the observation, reward, done state, and info
            ob, rew, done, info = self.env.step(action)
            
            # record the fitness, generic to all games
            fitness_current += rew
            
            # if the fitness is greater than the current max fitness, reset the frame counter and allow the episode to continue
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                frames = 0
            else:
                frames += 1
                
            if info['lives'] == 0 or frames > 250:
                # if the episode is done or 250 frames are recorded, end the episode
                done = True
            
            self.genome.fitness = fitness_current
                
        print(f"Fitness:{fitness_current}, {info['lives']}")
        return self.genome.fitness

def eval_genomes(genome, config):
    worker = Worker(genome, config)
    return worker.work()


if __name__ == '__main__':
    # load the config file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        '.config')
    # create the population
    p = neat.Population(config)

    # add reporters to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    pe = neat.ParallelEvaluator(num_workers=4, eval_function=eval_genomes)
    
    # n = number of generations
    winner = p.run(fitness_function=pe.evaluate, n=100)
    
    # save the winner
    winning_data = [winner, config, stats]
    with open('winner/winner.pkl', 'wb') as output:
        for data in winning_data:
            pickle.dump(data, output)
 