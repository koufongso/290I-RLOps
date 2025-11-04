import gymnasium as gym
import uuid
import numpy as np # Import numpy

class LunarLanderSimulator:
    def __init__(self, continuous: bool, gravity: float, enable_wind: bool, wind_power: float, turbulence_power: float):
        self.env = gym.make("LunarLander-v3", continuous=continuous, gravity=gravity,
                            enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
        
        self.id = str(uuid.uuid4()) 
        self.env_name = "LunarLander-v3" 

        # state variables
        self.state = None
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.info = {}

        # Configuration parameters
        self.continuous = continuous
        self.gravity = gravity
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        
        # Initialize state by resetting once
        self.reset()

    def reset(self):
        self.step_count = 0
        self.state, self.info = self.env.reset()
        return self.state, self.info

    def step(self, action: int):
        self.step_count += 1
        self.state, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        return self.state, self.reward, self.terminated, self.truncated, self.info

    def close(self):
        self.env.close()

    def to_json(self):
        serializable_state = None
        if isinstance(self.state, np.ndarray):
            serializable_state = self.state.tolist()
            
        serializable_info = {}

        return {
            "id": self.id,
            "environment": self.env_name,
            "state": serializable_state,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "step_count": self.step_count,
            "info": serializable_info,
            "config": {
                "continuous": self.continuous,
                "gravity": self.gravity,
                "enable_wind": self.enable_wind,
                "wind_power": self.wind_power,
                "turbulence_power": self.turbulence_power
            }
        }