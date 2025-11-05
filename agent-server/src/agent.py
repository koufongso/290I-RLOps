import uuid
import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from simulator_wrapper import LunarLanderSimulatorWrapper

class Agent:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.model = None
        self.status = "idle" # This state management is now perfect

    def train(self, simulator_id: str, simulator_environment: str, api_url: str):
        # self.status = "training" # This is now set by the server
        wrapper = None
        
        try:
            # set up the corresponding wrapper
            if simulator_environment == "LunarLander-v3":
                wrapper = LunarLanderSimulatorWrapper(
                    api_url=api_url, 
                    simulator_id=simulator_id
                )
            else:
                raise ValueError(f"Unknown simulator environment: {simulator_environment}")
            
            # create the model
            self.model = DQN("MlpPolicy", wrapper, verbose=0)

            print(f"Agent {self.id} start training...")
            self.model.learn(total_timesteps=int(2e5), progress_bar=True)
            print(f"Agent {self.id} finished training.")
            model_name = "DQN"
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            env_name = "LunarLander-V3"
            save_path = f"trained_models/{timestamp}_{model_name}_{env_name}.zip"
            self.model.save(save_path) # Save the model
            print(f"Agent {self.id} model saved.")

        finally:
            # clean up
            if wrapper:
                wrapper.close()
            self.status = "idle" # This is perfect!

    def predict(self, simulator_id: str, simulator_environment: str, api_url: str, eval_episodes: int = 100, load_path: str = None):
        wrapper = None # use a local variable
        
        try:
            # 1. set up the corresponding wrapper
            if simulator_environment == "LunarLander-v3":
                wrapper = LunarLanderSimulatorWrapper(
                    api_url=api_url, 
                    simulator_id=simulator_id
                )
            else:
                raise ValueError(f"Unknown simulator environment: {simulator_environment}")
                
            # 2. load the model (if specified)
            model_to_use = self.model
            if load_path:
                print(f"Loading model from {load_path}...")
                model_to_use = DQN.load(load_path, env=wrapper)
            
            if model_to_use is None:
                raise ValueError("Agent has no trained model. Either train or load a model.")

            print(f"Agent {self.id} start predicting...")
            mean_reward, std_reward = evaluate_policy(
                model_to_use, 
                wrapper, 
                n_eval_episodes=eval_episodes, 
                deterministic=True
            )
            print(f"Agent {self.id} finished predicting.")
            print(f"Mean reward: {mean_reward} +/- {std_reward}")

        finally:
            # clean up
            if wrapper:
                wrapper.close()
            self.status = "idle" # Also perfect!