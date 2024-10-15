import gym
import numpy as np

class MultiAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_agents = 3
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.grid_size = 50
        self.max_steps = 100
        self.current_step = 0
        self.agent_positions = None
        #self.target_position = None
        self.target_position = np.random.randint(0, self.grid_size, size=2)  # Set target position once

    def reset(self):
        self.current_step = 0
        self.agent_positions = [np.random.randint(0, self.grid_size, size=2) for _ in range(self.num_agents)]
        # uncomment to set a new target position
        # self.target_position = np.random.randint(0, self.grid_size, size=2)
        return np.array([self._get_observation(i) for i in range(self.num_agents)])

    def step(self, actions):
        self.current_step += 1
        rewards = []
        for i, action in enumerate(actions):
            if action == 0:  # up
                self.agent_positions[i][1] = min(self.grid_size - 1, self.agent_positions[i][1] + 1)
            elif action == 1:  # down
                self.agent_positions[i][1] = max(0, self.agent_positions[i][1] - 1)
            elif action == 2:  # left
                self.agent_positions[i][0] = max(0, self.agent_positions[i][0] - 1)
            elif action == 3:  # right
                self.agent_positions[i][0] = min(self.grid_size - 1, self.agent_positions[i][0] + 1)
            
            distance = np.linalg.norm(self.agent_positions[i] - self.target_position)
            rewards.append(-distance / self.grid_size)  # Normalize reward
        
        done = self.current_step >= self.max_steps or any(np.array_equal(pos, self.target_position) for pos in self.agent_positions)
        next_states = [self._get_observation(i) for i in range(self.num_agents)]
        info = {}
        return np.array(next_states), np.array(rewards), done, info

    def _get_observation(self, agent_id):
        return (self.agent_positions[agent_id] - self.target_position) / self.grid_size  # Normalize observation
