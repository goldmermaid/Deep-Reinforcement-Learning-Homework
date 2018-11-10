import tensorflow as tf
import numpy as np
import operator
import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        ## self._init_dataset: see utils.Dataset
        self._env = env
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self._state_dim], name='state_ph')
        action_ph = tf.placeholder(dtype=tf.float32, shape=[None, self._action_dim], name='action_ph')
        next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self._state_dim], name='next_state_ph')

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state                
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        ## (a) Normalize both the state and action by using the statistics of self._init_dataset and
        ##             the utils.normalize function
        state_mean = self._init_dataset.state_mean
        state_std = self._init_dataset.state_std
        state_normalize = utils.normalize(state, state_mean, state_std, eps=1e-8)

        # # @@@@@@ 
        # print("state", tf.convert_to_tensor(state).get_shape())
        # print("action", tf.convert_to_tensor(action).get_shape())
        # state (?, 20)
        # action (?, 6)
        action_mean = self._init_dataset.action_mean
        action_std = self._init_dataset.action_std
        action_normalize = utils.normalize(action, action_mean, action_std, eps=1e-8)

        ## (b) Concatenate the normalized state and action
        concatenated = tf.concat([state_normalize,action_normalize],axis=1,name='concatenated')

        # (c) Pass the concatenated, normalized state-action tensor through a neural network with
        #     self._nn_layers number of layers using the function utils.build_mlp. The resulting output
        #     is the normalized predicted difference between the next state and the current state
        next_state = utils.build_mlp(input_layer=concatenated, output_dim=self._state_dim, scope='dynamics_func', 
                                n_layers=self._nn_layers, reuse=reuse)

        # (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
        #     the predicted next state
        delta_state_mean = self._init_dataset.delta_state_mean
        delta_state_std = self._init_dataset.delta_state_std
        next_state_pred = state+utils.unnormalize(next_state, delta_state_mean, delta_state_std)
        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent                

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # (a) Compute both the actual state difference and the predicted state difference
        actual_state_diff = next_state_ph - state_ph
        pred_state_diff = next_state_pred - state_ph

        # (b) Normalize both of these state differences by using the statistics of self._init_dataset and
        #     the utils.normalize function
        state_mean = self._init_dataset.state_mean
        state_std = self._init_dataset.state_std
        actual_state_normalize = utils.normalize(actual_state_diff, state_mean, state_std, eps=1e-8)
        pred_state_normalize = utils.normalize(pred_state_diff, state_mean, state_std, eps=1e-8)

        # (c) The loss function is the mean-squared-error between the normalized state difference and
        #     normalized predicted state difference
        loss = tf.losses.mean_squared_error(labels=actual_state_normalize,predictions=pred_state_normalize)
        # (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)
        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])
        
        # Hints:
        # (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
        #     2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
        #     state or action size
        # (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
        # (iii) Use tf.random_uniform(...) to generate the random action sequences
        """

        ### PROBLEM 2
        ### YOUR CODE HERE
        # (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
        # (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
        #     each of length self._horizon

        # @@@@@@  print("self._init_dataset._actions[0]", tf.convert_to_tensor(self._init_dataset._actions[0]).get_shape())
        sampled_actions = tf.random_uniform(shape=[self._num_random_action_selection, self._horizon, self._action_dim],
                                            minval=self._action_space_low, maxval=self._action_space_high, dtype=tf.float32)        

        # (c) Starting from the input state, unroll each action sequence using your neural network
        #     dynamics model
        # (d) While unrolling the action sequences, keep track of the cost of each action sequence
        #     using self._cost_fn
        
        cost = 0  
        ## NOTE: the cost is a number rather than a list with len self._num_random_action_selection,
        ## since the env._cost_fn return a number, see details in half_cheetah_env.py
        for ac in range(self._horizon):
            action_ph = sampled_actions[:,ac,:]
            next_state_pred = self._dynamics_func(state_ph, action_ph, reuse=tf.AUTO_REUSE)   
            # !!!!!! reuse=False in utils.build_mlp
            cost += self._cost_fn(states=state_ph, actions=action_ph, next_states=next_state_pred)
            ## see details in half_cheetah_env.py : def cost_fn(states, actions, next_states):
            state_ph = next_state_pred

        # (e) Find the action sequence with the lowest cost, and return the first action in that sequence
        # min_index, min_value = min(enumerate(cost), key=operator.itemgetter(1))
        min_index = tf.argmin(cost)
        best_action = sampled_actions[min_index][0]
        # print("best_action", best_action.get_shape())

        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1 : The neural network dynamics model and train it using a fixed dataset 
        ###             consisting of rollouts collected by a random policy.
        ### YOUR CODE HERE
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, reuse= False) 
        ## !!!!!! reuse should be false here since each time setup_graph, we want a new network
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)
               

        ### PROBLEM 2 : Action selection using your learned dynamics model and a given cost function.
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, next_state_pred, loss, optimizer, best_action


    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE

        _,loss =self._sess.run([self._optimizer, self._loss],feed_dict={self._state_ph: states,
                                                                        self._action_ph: actions,
                                                                        self._next_state_ph: next_states})

        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        # print("self._state_dim ", tf.convert_to_tensor(self._state_dim).get_shape())
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        # sess, state_ph, action_ph, next_state_ph, next_state_pred, loss, optimizer, best_action = self._setup_graph()
        next_state_pred = self._sess.run([self._next_state_pred],feed_dict={self._state_ph: np.expand_dims(state,0),
                                                                        self._action_ph: np.expand_dims(action,0)})
        next_state_pred = np.squeeze(next_state_pred,axis=0)

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        ## convert the shape of state to [state, state, ...] with length self._num_random_action_selection
        current_state = np.asarray([state] * self._num_random_action_selection)
        best_action = self._sess.run([self._best_action],feed_dict={self._state_ph: current_state})
        best_action = np.squeeze(best_action,axis=0)

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
