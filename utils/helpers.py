import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from agents.agent import Agent
from environment.custom_env import MultiAgentEnv

def plot_rewards(all_rewards):
    plt.figure(figsize=(10, 5))
    
    # Get the 'Set1' colormap
    set1 = plt.cm.get_cmap('Set1')
    
    # Determine the number of agents
    num_agents = len(all_rewards)
    
    # Create a color list from 'Set1' based on the number of agents
    colors = plt.cm.Set1(np.linspace(0, 1, num_agents))
    
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f'Agent {i+1}', color=colors(i))
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agents\' Performance over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('agent_performance.png')
    plt.close()

def plot_metrics(all_metrics):
    metrics_to_plot = ['rewards', 'steps', 'epsilon', 'avg_q_value', 'loss', 'distance_to_target']
    num_agents = len(all_metrics[0]['rewards'])  # Determine the number of agents
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Training Metrics')

    # Generate a color map for the agents
    colors = plt.cm.Set1(np.linspace(0, 1, num_agents))

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i // 2, i % 2]
        if metric in ['rewards', 'epsilon', 'avg_q_value', 'loss', 'distance_to_target']:
            for agent in range(num_agents):
                ax.plot([m[metric][agent] for m in all_metrics], 
                        label=f'Agent {agent+1}', 
                        color=colors[agent])
        else:
            ax.plot([m[metric] for m in all_metrics])
        ax.set_title(metric.capitalize().replace('_', ' '))
        ax.set_xlabel('Episode')
        
        if i == 0:  # Only add legend to the first subplot to save space
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig('training_metrics.png', bbox_inches='tight')
    plt.close()

def plot_grid(env, agent_paths, episode):
    plt.figure(figsize=(12, 12))
    plt.title(f"Agent Paths - Episode {episode}")
    plt.xlim(0, env.grid_size - 1)
    plt.ylim(0, env.grid_size - 1)
    plt.grid(True)

    # Plot target
    plt.plot(env.target_position[0], env.target_position[1], 'r*', markersize=20, label='Target')

    # Generate a color map for the agents
    num_agents = len(agent_paths)
    colors = plt.cm.Set1(np.linspace(0, 1, num_agents))

    # Plot agent paths
    for i, path in enumerate(agent_paths):
        path = np.array(path)
        if len(path) > 0:
            plt.plot(path[1:, 0], path[1:, 1], c=colors[i], linestyle='-', label=f'Agent {i+1} Path')
            plt.plot(path[1, 0], path[1, 1], c=colors[i], marker='o', markersize=10, label=f'Agent {i+1} Start')
            plt.plot(path[-1, 0], path[-1, 1], c=colors[i], marker='s', markersize=10, label=f'Agent {i+1} End')

    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.savefig(f'agent_paths_episode_{episode}.png', bbox_inches='tight')
    plt.close()

def track_metrics(env, agents, episode, episode_rewards, steps, agent_paths):
    metrics = {
        'episode': episode,
        'rewards': episode_rewards,
        'steps': steps,
        'epsilon': [agent.epsilon for agent in agents],
        'avg_q_value': [],
        'loss': [],
        'distance_to_target': []
    }

    # Calculate average Q-value and loss
    for agent in agents:
        state = env.reset()[0]  # Get a sample state
        state = torch.FloatTensor(np.array([state])).to(agent.device)  # Convert to numpy array first
        q_values = agent.policy_net(state).mean().item()
        metrics['avg_q_value'].append(q_values)

        if len(agent.memory) > 64:
            batch = random.sample(agent.memory, 64)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to numpy arrays first, then to tensors
            states = torch.FloatTensor(np.array(states)).to(agent.device)
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(agent.device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(agent.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(agent.device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(agent.device)
            
            current_q_values = agent.policy_net(states).gather(1, actions)
            next_q_values = agent.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (agent.gamma * next_q_values * (1 - dones))
            
            loss = nn.MSELoss()(current_q_values, target_q_values)
            metrics['loss'].append(loss.item())
        else:
            metrics['loss'].append(0)

    # Calculate distance to target
    for pos in env.agent_positions:
        distance = np.linalg.norm(pos - env.target_position)
        metrics['distance_to_target'].append(distance)

    metrics['agent_paths'] = agent_paths

    return metrics

def load_trained_agents(model_path, num_agents):
    env = MultiAgentEnv()
    agents = [Agent(env.observation_space.shape[0], env.action_space.n, i) for i in range(num_agents)]
    for i, agent in enumerate(agents):
        agent.load_model(model_path)
    return agents

