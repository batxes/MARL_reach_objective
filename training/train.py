from utils.helpers import track_metrics, plot_grid, plot_metrics, plot_rewards
from environment.custom_env import MultiAgentEnv
from agents.agent import Agent

def train(num_episodes=1000, save_interval=100):
    env = MultiAgentEnv()
    agents = [Agent(env.observation_space.shape[0], env.action_space.n, i) for i in range(env.num_agents)]

    all_rewards = [[] for _ in range(env.num_agents)]
    all_metrics = []

    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = [0] * env.num_agents
        done = False
        steps = 0
        agent_paths = [[] for _ in range(env.num_agents)]

        # Record initial positions
        for i in range(env.num_agents):
            agent_paths[i].append(env.agent_positions[i])

        while not done:
            actions = [agent.choose_action(state) for agent, state in zip(agents, states)]
            next_states, rewards, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.learn(states[i], actions[i], rewards[i], next_states[i], done)
                episode_rewards[i] += rewards[i]
                agent_paths[i].append(env.agent_positions[i].copy())  # Append a copy of the current position

            states = next_states
            steps += 1

        for i in range(env.num_agents):
            all_rewards[i].append(episode_rewards[i])
        
        metrics = track_metrics(env, agents, episode, episode_rewards, steps, agent_paths)
        all_metrics.append(metrics)

        if episode % 100 == 0:
            print(f"Episode {episode}, Rewards: {episode_rewards}")
            plot_grid(env, agent_paths, episode)

        if (episode + 1) % save_interval == 0:
            for agent in agents:
                agent.save_model(f'saved_models/episode_{episode+1}')

    # Save final models
    for agent in agents:
        agent.save_model('saved_models/final')

    plot_rewards(all_rewards)
    plot_metrics(all_metrics)

    return agents, all_rewards
