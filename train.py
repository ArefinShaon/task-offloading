from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

def reward_fun(delay, max_delay, unfinish_indi):
    penalty = -max_delay * (1 + np.tanh(0.1 * delay))  
    completion_reward = 20 if not unfinish_indi else 0  
    scaled_delay = delay / max_delay
    shaping_bonus = 10 / (1 + abs(scaled_delay - 0.5))  
    reward = -np.log1p(scaled_delay) + completion_reward + shaping_bonus if not unfinish_indi else penalty
    reward = np.clip(reward, -10, 30)  
    return reward




def train(iot_RL_list, NUM_EPISODE):
    RL_step = 0
    total_rewards = []
    dropped_tasks_ratio = []
    average_delay_list = []

    for episode in range(NUM_EPISODE):
        task_prob = 0.1 + (episode / (NUM_EPISODE - 1)) * 0.5  
        print(f'Episode: {episode}, Task arrival Probability: {task_prob:.2f}')

        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = 0

        history = [[{'observation': np.zeros(env.n_features),
                     'lstm': np.zeros(env.n_lstm_state),
                     'action': np.nan,
                     'observation_': np.zeros(env.n_features),
                     'lstm_': np.zeros(env.n_lstm_state)}
                    for _ in range(env.n_iot)] for _ in range(env.n_time)]

        reward_indicator = np.zeros([env.n_time, env.n_iot])
        observation_all, lstm_state_all = env.reset(bitarrive)
        total_episode_reward = 0
        total_dropped_tasks = 0
        total_tasks = 0
        total_delay = 0
        total_completed_tasks = 0

        while True:
            action_all = np.zeros(env.n_iot)
            for iot_index in range(env.n_iot):
                observation = np.squeeze(observation_all[iot_index, :])
                action_all[iot_index] = 0 if np.sum(observation) == 0 else iot_RL_list[iot_index].choose_action(observation)
                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            observation_all_, lstm_state_all_, done = env.step(action_all)
            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index,:])

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind
            total_tasks += np.sum(bitarrive > 0)
            total_dropped_tasks += np.sum(unfinish_indi)
            completed_task_delays = process_delay[process_delay > 0]
            total_delay += np.sum(completed_task_delays)
            total_completed_tasks += len(completed_task_delays)

            for iot_index in range(env.n_iot):
                history[env.time_count - 1][iot_index].update({
                    'observation': observation_all[iot_index, :],
                    'lstm': np.squeeze(lstm_state_all[iot_index, :]),
                    'action': action_all[iot_index],
                    'observation_': observation_all_[iot_index],
                    'lstm_': np.squeeze(lstm_state_all_[iot_index, :])
                })
                update_index = np.where((1 - reward_indicator[:,iot_index]) * process_delay[:,iot_index] > 0)[0]
                for time_index in update_index:
                    reward = reward_fun(process_delay[time_index, iot_index], env.max_delay, unfinish_indi[time_index, iot_index])
                    iot_RL_list[iot_index].store_transition(
                        history[time_index][iot_index]['observation'],
                        history[time_index][iot_index]['lstm'],
                        history[time_index][iot_index]['action'],
                        reward,
                        history[time_index][iot_index]['observation_'],
                        history[time_index][iot_index]['lstm_']
                    )
                    iot_RL_list[iot_index].do_store_reward(episode, time_index, reward)
                    iot_RL_list[iot_index].do_store_delay(episode, time_index, process_delay[time_index, iot_index])
                    reward_indicator[time_index, iot_index] = 1
                    total_episode_reward += reward

            RL_step += 1
            observation_all, lstm_state_all = observation_all_, lstm_state_all_

            if (RL_step > 50) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            if done:
                break

        avg_delay = total_delay / (total_completed_tasks + 1e-6)
        avg_delay_normalized = avg_delay / env.max_delay
        dropped_tasks_ratio.append(total_dropped_tasks / total_tasks if total_tasks > 0 else 0)
        average_delay_list.append(avg_delay_normalized)
        total_rewards.append(total_episode_reward)
        print(f'Total Episode Reward: {total_episode_reward}')

    plt.figure()
    plt.plot((dropped_tasks_ratio), 'o-', label='DRL')
    plt.xlabel('Task Arrival Probability')
    plt.ylabel('Ratio of Dropped Tasks')
    plt.title('DRL: Ratio of Dropped Tasks vs Task Arrival Probability')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot((average_delay_list), 'o-', label='DRL')
    plt.xlabel('Task Arrival Probability')
    plt.ylabel('Normalized Average Delay (Ratio)')
    plt.title('DRL: Normalized Average Delay vs Task Arrival Probability')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot((total_rewards))
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episode')
    plt.show()

if __name__ == "__main__":
    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 100
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    iot_RL_list = list()
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=0.001,
                                        reward_decay=0.99,
                                        e_greedy=0.95,
                                        replace_target_iter=500,
                                        memory_size=1000,
                                        batch_size=32))

    train(iot_RL_list, NUM_EPISODE)
    print('Training Finished')