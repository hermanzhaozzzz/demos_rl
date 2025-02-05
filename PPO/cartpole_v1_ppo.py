import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ------------------ 1. 定义模型 ------------------
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# ------------------ 2. 定义 PPO 训练类 ------------------
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.checkpoint_dir = 'checkpoints'
        self._saved_pth = ''
        self.episode_counter = 0 

    def save_checkpoint(self, save_interval=100):
        """ 保存 Actor 和 Critic 网络权重 """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._saved_pth = os.path.join(self.checkpoint_dir, 'checkpoints' + str(self.episode_counter) + '.pth')
                                       
        torch.save(checkpoint, self._saved_pth) 
        print(f"Checkpoint saved to {self._saved_pth}")

    def load_checkpoint(self, checkpoint=None):
        """ 从 checkpoint 加载权重 """
        if checkpoint and os.path.exists(checkpoint):
            pth = checkpoint
        else:
            pth = self._saved_pth
        checkpoint = torch.load(pth, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Checkpoint loaded from {checkpoint}")

    def take_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, td_delta):
        advantage = []
        acc = 0
        for delta in reversed(td_delta.detach().cpu().numpy()):
            acc = delta + self.gamma * self.lmbda * acc
            advantage.insert(0, acc)
        return torch.tensor(np.array(advantage), dtype=torch.float).to(self.device)

    def update(self, transition_dict):
        self.episode_counter += 1
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(td_delta).detach()

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        actor_loss_list = []
        critic_loss_list = []

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())

        return actor_loss_list, critic_loss_list

# ------------------ 3. 训练和测试 ------------------
def train_on_policy_agent(env, agent, num_episodes, save_interval=50):
    return_list = []
    actor_loss_list = []
    critic_loss_list = []

    for i in range(num_episodes):
        transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        state, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

            episode_return += reward
            state = next_state

        return_list.append(episode_return)
        actor_loss, critic_loss = agent.update(transition_dict)
        actor_loss_list.extend(actor_loss)
        critic_loss_list.extend(critic_loss)

        if (i + 1) % save_interval == 0:
            agent.save_checkpoint(save_interval)
            print(f'Episode {i+1}, Return: {episode_return}')

    return return_list, actor_loss_list, critic_loss_list


def test_agent(env, agent, num_episodes=10, render=False):
    """
    测试 PPO 训练好的 agent，在环境中运行 num_episodes 轮，并计算平均奖励。
    
    参数:
    - env: Gym 环境
    - agent: 训练好的 PPO 代理
    - num_episodes: 测试轮数
    - render: 是否渲染环境（默认 False，加速测试）

    返回:
    - avg_return: agent 在测试过程中的平均奖励
    """
    total_return = 0

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            if render:
                env.render()  # 渲染环境，可视化
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            state = next_state

        total_return += episode_return
        print(f'Test Episode {i+1}, Return: {episode_return}')

    avg_return = total_return / num_episodes
    print(f'Average Return over {num_episodes} episodes: {avg_return:.2f}')
    return avg_return

# ------------------ 4. 运行 ------------------
env_name = "CartPole-v1"
actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.99
lmbda = 0.97
epochs = 4
eps = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_checkpoint = ''  # 是否加载已有模型

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
if load_checkpoint:
    agent.load_checkpoint(load_checkpoint)

return_list, actor_loss_list, critic_loss_list = train_on_policy_agent(env, agent, num_episodes)

# ------------------ 5. 测试 ------------------
test_agent(env, agent, num_episodes=10, render=False)



# ------------------ 6. 结果可视化 ------------------
def smooth(data, weight=0.9):
    smoothed = []
    last = data[0]
    for point in data:
        last = last * weight + (1 - weight) * point
        smoothed.append(last)
    return smoothed

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(return_list, label='rewards')
plt.plot(smooth(return_list), label='rewards (smoothed)')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.legend()
plt.title('PPO on {}'.format(env_name))

plt.subplot(1, 2, 2)
plt.plot(smooth(actor_loss_list), label="Actor Loss")
plt.plot(smooth(critic_loss_list), label="Critic Loss")
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Time')

plt.show()
