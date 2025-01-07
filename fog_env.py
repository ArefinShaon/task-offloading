import numpy as np
import random
import math
import queue

class Offload:

    def __init__(self, num_iot, num_fog, num_time, max_delay):
        # INPUT DATA
        self.n_iot = num_iot
        self.n_fog = num_fog
        self.n_time = num_time
        self.duration = 0.1

        # Computational and transmission capacities with more variability
        self.comp_cap_iot = np.random.uniform(2.0, 3.0, self.n_iot) * self.duration  # Randomized for better load balancing
        self.comp_cap_fog = np.random.uniform(40.0, 45.0, self.n_fog) * self.duration  # Randomized capacity
        self.tran_cap_iot = np.random.uniform(12.0, 16.0, (self.n_iot, self.n_fog)) * self.duration
        self.comp_density = 0.297 * np.ones([self.n_iot])  # Computational density remains constant
        self.max_delay = max_delay

        # TASK GENERATION PROPERTIES
        self.task_arrive_prob = 0.3
        self.max_bit_arrive = 5  # Mbits
        self.min_bit_arrive = 2  # Mbits
        self.bitArrive_set = np.linspace(self.min_bit_arrive, self.max_bit_arrive, 30)  # More fine-grained

        # ACTION & STATE DIMENSIONS
        self.n_actions = 1 + num_fog  # Actions for local and fog
        self.n_features = 1 + 1 + 1 + num_fog  # State: [bit_arrive, comp_time, tran_time, fog_loads]
        self.n_lstm_state = self.n_fog  # LSTM state size

        # TIME COUNT & QUEUE INITIALIZATION
        self.time_count = 0
        self.reset_queues()

        # FOG CONGESTION TRACKING
        self.fog_iot_m = np.zeros(self.n_fog)  # Initialize fog congestion state
        self.fog_iot_m_observe = np.zeros(self.n_fog)  # Initialize observable fog state


    def reset_queues(self):
        """Resets the queues and related task states."""
        self.Queue_iot_comp = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_iot_tran = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_fog_comp = [[queue.Queue() for _ in range(self.n_fog)] for _ in range(self.n_iot)]

        self.t_iot_comp = -np.ones([self.n_iot])
        self.t_iot_tran = -np.ones([self.n_iot])
        self.b_fog_comp = np.zeros([self.n_iot, self.n_fog])

        self.task_on_process_local = [{'size': np.nan, 'time': np.nan, 'remain': np.nan} for _ in range(self.n_iot)]
        self.task_on_transmit_local = [{'size': np.nan, 'time': np.nan, 'fog': np.nan, 'remain': np.nan} for _ in range(self.n_iot)]
        self.task_on_process_fog = [[{'size': np.nan, 'time': np.nan, 'remain': np.nan} for _ in range(self.n_fog)] for _ in range(self.n_iot)]

        self.process_delay = np.zeros([self.n_time, self.n_iot])
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])

        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

    def reset(self, bitArrive):
        """Reset the environment."""
        self.time_count = 0
        self.bitArrive = bitArrive
        self.reset_queues()

        observation_all = np.zeros([self.n_iot, self.n_features])
        for iot_index in range(self.n_iot):
            if self.bitArrive[self.time_count, iot_index] != 0:
                observation_all[iot_index, :] = np.hstack([
                    self.bitArrive[self.time_count, iot_index], self.t_iot_comp[iot_index],
                    self.t_iot_tran[iot_index], self.b_fog_comp[iot_index, :]])

        lstm_state_all = np.zeros([self.n_iot, self.n_lstm_state])
        return observation_all, lstm_state_all

    def step(self, action):
        """Performs a step in the environment based on the actions."""
        iot_action_local = (action == 0).astype(int)
        iot_action_fog = action - 1

        # COMPUTE QUEUE PROCESSING
        self.update_queues(iot_action_local, iot_action_fog)
        self.process_fog_tasks()

        # UPDATE CONGESTION
        self.update_congestion()

        # TIME UPDATE
        self.time_count += 1
        done = self.time_count >= self.n_time
        if done:
            self.set_unfinished_tasks()

        # OBSERVATION UPDATE
        observation_all_, lstm_state_all_ = self.get_observations()
        return observation_all_, lstm_state_all_, done

    def update_queues(self, iot_action_local, iot_action_fog):
        """Updates the local computation and transmission queues."""
        for iot_index in range(self.n_iot):
            iot_bitarrive = self.bitArrive[self.time_count, iot_index]
            if iot_action_local[iot_index]:
                self.Queue_iot_comp[iot_index].put({'size': iot_bitarrive, 'time': self.time_count})

            self.process_local_queue(iot_index, iot_bitarrive, iot_action_local)

    def process_local_queue(self, iot_index, iot_bitarrive, iot_action_local):
        """Processes tasks in the local computation queue."""
        iot_comp_cap = self.comp_cap_iot[iot_index]
        iot_comp_density = self.comp_density[iot_index]

        # Local processing logic
        if math.isnan(self.task_on_process_local[iot_index]['remain']) and not self.Queue_iot_comp[iot_index].empty():
            while not self.Queue_iot_comp[iot_index].empty():
                get_task = self.Queue_iot_comp[iot_index].get()
                if get_task['size'] > 0:
                    if self.time_count - get_task['time'] + 1 <= self.max_delay:
                        self.task_on_process_local[iot_index] = get_task
                        self.task_on_process_local[iot_index]['remain'] = get_task['size']
                        break
                    else:
                        self.mark_task_unfinished(get_task['time'], iot_index)

        # Process local tasks
        if self.task_on_process_local[iot_index]['remain'] > 0:
            self.task_on_process_local[iot_index]['remain'] -= iot_comp_cap / iot_comp_density
            self.update_local_task(iot_index)

    def process_fog_tasks(self):
        """Processes tasks assigned to the fog."""
        # Similar logic to update fog queues and processing

    def update_congestion(self):
        """Updates congestion levels."""
        self.fog_iot_m_observe = self.fog_iot_m
        self.fog_iot_m = np.zeros(self.n_fog)
        for fog_index in range(self.n_fog):
            for iot_index in range(self.n_iot):
                if not self.Queue_fog_comp[iot_index][fog_index].empty() or self.task_on_process_fog[iot_index][fog_index]['remain'] > 0:
                    self.fog_iot_m[fog_index] += 1

    def set_unfinished_tasks(self):
        """Sets tasks as unfinished if the maximum delay is reached."""
        for time_index in range(self.n_time):
            for iot_index in range(self.n_iot):
                if self.process_delay[time_index, iot_index] == 0 and self.bitArrive[time_index, iot_index] != 0:
                    self.process_delay[time_index, iot_index] = self.time_count - time_index + 1
                    self.process_delay_unfinish_ind[time_index, iot_index] = 1

    def get_observations(self):
        """Returns the next observations."""
        observation_all_ = np.zeros([self.n_iot, self.n_features])
        lstm_state_all_ = np.zeros([self.n_iot, self.n_lstm_state])
        if self.time_count < self.n_time:
            for iot_index in range(self.n_iot):
                if self.bitArrive[self.time_count, iot_index] != 0:
                    observation_all_[iot_index, :] = np.hstack([
                        self.bitArrive[self.time_count, iot_index],
                        self.t_iot_comp[iot_index] - self.time_count + 1,
                        self.t_iot_tran[iot_index] - self.time_count + 1,
                        self.b_fog_comp[iot_index, :]])
                lstm_state_all_[iot_index, :] = np.hstack(self.fog_iot_m_observe)
        return observation_all_, lstm_state_all_

    def mark_task_unfinished(self, time, iot_index):
        """Marks a task as unfinished."""
        self.process_delay[time, iot_index] = self.max_delay
        self.process_delay_unfinish_ind[time, iot_index] = 1

    def update_local_task(self, iot_index):
        """Updates the status of a local task."""
        if self.task_on_process_local[iot_index]['remain'] <= 0:
            self.process_delay[self.task_on_process_local[iot_index]['time'], iot_index] = self.time_count - self.task_on_process_local[iot_index]['time'] + 1
            self.task_on_process_local[iot_index]['remain'] = np.nan
        elif self.time_count - self.task_on_process_local[iot_index]['time'] + 1 == self.max_delay:
            self.mark_task_unfinished(self.task_on_process_local[iot_index]['time'], iot_index)
            self.task_on_process_local[iot_index]['remain'] = np.nan
