import sys
import os
from utils.helpers import load_trained_agents
from environment.custom_env import MultiAgentEnv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from training.train import train

def evaluate(num_episodes=100):
    env = MultiAgentEnv()
    agents = load_trained_agents('saved_models/final', env.num_agents)
    
    for episode in range(num_episodes):
        states = env.reset()
        done = False
        total_rewards = [0] * env.num_agents

        while not done:
            actions = [agent.choose_action(state) for agent, state in zip(agents, states)]
            next_states, rewards, done, _ = env.step(actions)
            
            for i in range(env.num_agents):
                total_rewards[i] += rewards[i]

            states = next_states

        print(f"Episode {episode+1}, Total Rewards: {total_rewards}")

def main():
    trained_agents, rewards = train(num_episodes=1000)
    print("Training completed.")

    # Evaluate trained agents
    evaluate()
    print ("Evaluation completed.")

if __name__ == "__main__":
    main()