import collections
import random

import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []  # 存储每回合的总回报
    for i in range(10):  # 将训练分为 10 个阶段
        with tqdm(
            total=int(num_episodes / 10), desc="Iteration %d" % i
        ) as pbar:  # 创建进度条
            for i_episode in range(
                int(num_episodes / 10)
            ):  # 每阶段训练 num_episodes/10 回合
                episode_return = 0  # 初始化当前回合的总回报
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }  # 记录当前回合的数据
                state, _ = env.reset()  # 重置环境 # 返回的是初始状态和额外信息
                done = False
                while not done:  # 游戏未结束时继续
                    action = agent.take_action(state)  # 选择动作

                    # fix gymnasium difference with gym
                    # next_state, reward, done, _ = env.step(action)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 执行动作，获取环境反馈
                    # //fix gymnasium difference with gym

                    transition_dict["states"].append(state)  # 记录状态
                    transition_dict["actions"].append(action)  # 记录动作
                    transition_dict["next_states"].append(next_state)  # 记录下一状态
                    transition_dict["rewards"].append(reward)  # 记录奖励
                    transition_dict["dones"].append(done)  # 记录是否结束

                    state = next_state  # 更新当前状态
                    episode_return += reward  # 累加回报
                return_list.append(episode_return)  # 记录当前回合总回报
                agent.update(transition_dict)  # 使用当前回合数据更新策略
                if (i_episode + 1) % 10 == 0:  # 每 10 个回合更新进度条信息
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)  # 更新进度条
    return return_list  # 返回所有回合的总回报


def train_off_policy_agent(
    env, agent, num_episodes, replay_buffer, minimal_size, batch_size
):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)

                    # fix gymnasium difference with gym
                    # next_state, reward, done, _ = env.step(action)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 执行动作，获取环境反馈
                    # //fix gymnasium difference with gym

                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()  # 将 TD-误差转换为 NumPy 数组
    advantage_list = []  # 初始化优势值列表
    advantage = 0.0  # 初始化递归变量
    for delta in td_delta[::-1]:  # 从后往前遍历 TD-误差
        advantage = gamma * lmbda * advantage + delta  # 递归计算优势值
        advantage_list.append(advantage)  # 存储优势值
    advantage_list.reverse()  # 恢复正序
    return torch.from_numpy(
        np.array(advantage_list, dtype=np.float32)
    )  # 返回优势值张量
