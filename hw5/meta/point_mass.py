import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1):
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False, is_generalization=False, p=10.0):

        '''
        input:
        is_generalization: whether using checkboard or not
        p: how many segments_from_0_to_10


        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        if not is_generalization:
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            self._goal = np.array([x, y]) 

        else:
            # ## idea 1: 
            ## set the upper triangle of a square to be evaluation set, lower triangle of a square to be training set

            # ## To be specific, if we want to sample from training set,
            # ## within the square with length one that contains point (x,y), if the point is in the lower triangle 
            # ## (ie.the x_excessive > y_excessive), then clearify as training set. Otherwise, if the point is in the upper triangle,
            # ## flip the point to corresponsing lower triangle (ie. (np.ceil(x) - y_excessive, np.ceil(y)-x_excessive)
            # x_excessive = x-np.floor(x)
            # y_excessive = y-np.floor(y)
            # if (is_evaluation & (x_excessive > y_excessive)) | (not is_evaluation & (x_excessive < y_excessive)):
            #     x_update = np.ceil(x) - y_excessive
            #     y_update = np.ceil(y) - x_excessive
            #     self._goal = np.array([x_update, y_update])
            # else:
            #     self._goal = np.array([x, y]) 


            ## idea 2: 
            ## first sample uniformly from ([-a,a],[-a,a]), i.e. checkboard square with length 2a and split into 4 squares.
            ## take the first and third quadrants as training set, the second and fourth quadrants as evaluating set. 
            ## If we need a sample from training sets, but draw a point in the second and fourth quadrants, then flip the point 
            ## across x axis, (i.e. dot porduct (-1,1)); similarly for sampling from evaluating sets.

            square_length = 10.0/p
            x = np.random.uniform(-1*square_length, square_length)
            y = np.random.uniform(-1*square_length, square_length)
            if (is_evaluation & (x*y >0)) | ((not is_evaluation) & (x*y<0)):
                x = -x
            ## Then move the "adjusted point" to the lower left corner of the ([-10,10],[-10,10]) matrix
            x -= (10-square_length)
            y -= (10-square_length)

            ## Finally, add length of {randint([0,1,...,p-1]) * 2 * square_length} to x and y, to generalize to the ([-10,10],[-10,10]) matrix
            x += np.random.randint(p) * 2 * square_length
            y += np.random.randint(p) * 2 * square_length

            self._goal = np.array([x, y]) 

        

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state.flatten()
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
