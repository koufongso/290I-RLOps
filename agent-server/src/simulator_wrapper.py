import gymnasium as gym
from gymnasium import spaces
import httpx
import numpy as np

class LunarLanderSimulatorWrapper(gym.Env):
    """
    A custom Gym environment that interacts with the Simulator Server API.
    """
    def __init__(self, api_url: str, simulator_id: str):
        super(LunarLanderSimulatorWrapper, self).__init__() 
        
        self.client = httpx.Client(base_url=api_url)
        self.simulator_id = simulator_id
        
        # Define action and observation space for LunarLander
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def step(self, action):
        response = self.client.post(
            f"/simulators/{self.simulator_id}/step",
            json={"action": int(action)}
        )
        response.raise_for_status()
        data = response.json()

        state = np.array(data["state"], dtype=np.float32)
        reward = data["reward"]
        terminated = data["terminated"]
        truncated = data["truncated"]
        info = data["info"]
        
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # This will now work
        response = self.client.post(f"/simulators/{self.simulator_id}/reset")
        response.raise_for_status()
        data = response.json()

        state = np.array(data["state"], dtype=np.float32)
        info = data["info"]
        
        return state, info

    def close(self):
        # Close the HTTP client
        self.client.close()