import numpy as np
import tensorflow as tf
from collections import deque

class DeepQNetwork:
    def __init__(self, n_actions, n_features, n_lstm_features, n_time, learning_rate=0.001, reward_decay=0.99,
                 e_greedy=0.95, replace_target_iter=500, memory_size=1000, batch_size=32, e_greedy_increment=None,
                 n_lstm_step=10, dueling=True, double_q=True, N_L1=32, N_lstm=32):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.N_L1 = N_L1
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features
        self.memory = deque(maxlen=self.memory_size)

        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.reward_store = []
        self.action_store = []
        self.delay_store = []

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

    def _build_net(self):
        tf.reset_default_graph()

        def build_layers(s, lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer):
            with tf.variable_scope('l0'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_lstm)
                lstm_output, _ = tf.nn.dynamic_rnn(lstm_cell, lstm_s, dtype=tf.float32)
                lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [n_lstm + self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_output_reduced, s], 1), w1) + b1)

            if self.dueling:
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2
                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.lstm_s = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name='lstm1_s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.lstm_s_ = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name='lstm1_s_')

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = build_layers(self.s, self.lstm_s, c_names, self.N_L1, self.N_lstm,
                                       tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1))

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.lstm_s_, c_names, self.N_L1, self.N_lstm,
                                       tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1))

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if len(self.memory) >= self.memory_size:
            self.memory.popleft()
        self.memory.append((s, lstm_s, a, r, s_, lstm_s_))

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            lstm_observation = np.array(self.lstm_history)
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation,
                                                                  self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def pad_or_truncate_lstm_sequences(self, sequences, n_lstm_step, n_lstm_state):
        adjusted_sequences = []
        for seq in sequences:
            if len(seq) < n_lstm_step:
                padded = np.zeros((n_lstm_step, n_lstm_state))
                padded[:len(seq), :] = seq
                adjusted_sequences.append(padded)
            else:
                adjusted_sequences.append(seq[:n_lstm_step])
        return np.array(adjusted_sequences)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if len(self.memory) < self.batch_size:
            print("Not enough samples to learn. Skipping learning step.")
            return

        sample_index = np.random.choice(len(self.memory), self.batch_size)
        batch_memory = [self.memory[idx] for idx in sample_index]

        s, lstm_s, a, r, s_, lstm_s_ = zip(*batch_memory)
        s = np.array(s)
        s_ = np.array(s_)
        lstm_s = self.pad_or_truncate_lstm_sequences(lstm_s, self.n_lstm_step, self.n_lstm_state)
        lstm_s_ = self.pad_or_truncate_lstm_sequences(lstm_s_, self.n_lstm_step, self.n_lstm_state)

        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: s_, self.lstm_s_: lstm_s_, self.s: s, self.lstm_s: lstm_s})

        q_eval = self.sess.run(self.q_eval, {self.s: s, self.lstm_s: lstm_s})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = np.array(a, dtype=np.int32)  # Ensure integer type
        reward = np.array(r)

        if self.double_q:
             max_act4next = np.argmax(q_eval4next, axis=1)
             selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: s, self.lstm_s: lstm_s, self.q_target: q_target})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def do_store_action(self, episode, time_index, action):
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time_index] = action

    def do_store_delay(self, episode, time_index, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time_index] = delay

    def update_lstm(self, lstm_state):
        self.lstm_history.append(lstm_state)

    def do_store_reward(self, episode, time_index, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time_index] = reward
