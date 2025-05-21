import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
 
class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        # start epsilon at 0 if we will increment, else at max
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
 
        self.learn_step_counter = 0
        # memory: [s, a, r, s_], so width = n_features * 2 + 2
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
 
        # build eval & target nets
        self._build_net()
 
        # replace target parameters op
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluation_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)
            ]
 
        # session & init
        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
 
    def _build_net(self):
        # ----- placeholders -----
        self.s   = tf.placeholder(tf.float32, [None, self.n_features], name='s')    # current state
        self.s_  = tf.placeholder(tf.float32, [None, self.n_features], name='s_')   # next state
        self.r   = tf.placeholder(tf.float32, [None, ], name='r')                  # reward
        self.a   = tf.placeholder(tf.int32,   [None, ], name='a')                  # action taken
 
        w_init = tf.random_normal_initializer(0., 0.3)
        b_init = tf.constant_initializer(0.1)
 
        # ----- evaluation network -----
        with tf.variable_scope('evaluation_net'):
            e1 = tf.layers.dense(
                self.s, 20, tf.nn.relu,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='e1'
            )
            self.q_eval = tf.layers.dense(
                e1, self.n_actions,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='q'
            )
 
        # ----- target network -----
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(
                self.s_, 20, tf.nn.relu,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='t1'
            )
            self.q_next = tf.layers.dense(
                t1, self.n_actions,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='t2'
            )
 
        # ----- Q target -----
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
 
        # ----- Q eval for the taken actions -----
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],
                axis=1
            )
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
 
        # ----- loss & train -----
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval_wrt_a),
                name='TD_error'
            )
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
 
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
 
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            q_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
 
    def learn(self):
        # update target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\n--- target network parameters replaced ---\n')
 
        # sample batch from memory
        if self.memory_counter > self.memory_size:
            sample_idx = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_idx = np.random.choice(self.memory_counter, size=self.batch_size)
        batch = self.memory[sample_idx, :]
 
        # train
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s:   batch[:, :self.n_features],
                self.a:   batch[:, self.n_features],
                self.r:   batch[:, self.n_features + 1],
                self.s_:  batch[:, -self.n_features:]
            }
        )
        self.cost_his.append(cost)
 
        # increase epsilon
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
 
        self.learn_step_counter += 1