import uuid
import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from simulator_wrapper import LunarLanderSimulatorWrapper

class Agent:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.model = None
        self.status = "idle" # This state management is now perfect
        self.status_timestamp = datetime.datetime.now()
        self.result_message = None  # result message for train/predict if successful, need to be reset before each call
        self.error_message = None   # error message for train/predict if failed, need to be reset before each call

    def train(self, simulator_id: str, simulator_environment: str, api_url: str, total_timesteps: int = 20000, filename: str = None):
        # self.status = "training" # This is now set by the server
        wrapper = None
        self.error_message = None
        self.result_message = None
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
            self.model = DQN("MlpPolicy", wrapper, verbose=1)
            wrapper.reset() # reset the env before training
            print(f"Agent {self.id} start training...")
            self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
            print(f"Agent {self.id} finished training.")
            
            # save model
            model_name = "DQN"
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            env_name = "LunarLander-V3"
            if filename is None:
                filename = f"{timestamp}_{model_name}_{env_name}"

            file_path = f"trained_models/{filename}.zip"
            self.model.save(file_path) # Save the model
            print(f"Agent {self.id} model saved.")
            self.result_message = f"Model saved as {filename}"

        except Exception as e:
            print(f"Error occurred during training: {e}")
            self.error_message = str(e)
        finally:
            # clean up
            if wrapper:
                wrapper.close()
            self.update_status("idle")

    def predict(self, simulator_id: str, simulator_environment: str, api_url: str, eval_episodes: int = 10, save_filename: str = None):
        wrapper = None # use a local variable
        self.error_message = None
        self.result_message = None
        try:
            # 1. set up the corresponding wrapper
            if simulator_environment == "LunarLander-v3":
                wrapper = LunarLanderSimulatorWrapper(
                    api_url=api_url, 
                    simulator_id=simulator_id
                )
                wrapper = Monitor(wrapper) # for the evaluate_policy 
            else:
                raise ValueError(f"Unknown simulator environment: {simulator_environment}")
                
            # 2. load the model (if specified)
            model_to_use = self.model
            if save_filename:
                print(f"Loading model from {save_filename}...")
                load_path = f"./trained_models/{save_filename}.zip"
                model_to_use = DQN.load(load_path, env=wrapper)

            if model_to_use is None:
                raise ValueError("Agent has no trained model. Either train or load a model.")

            wrapper.reset() # reset the env before predicting
            print(f"Agent {self.id} start predicting...")
            mean_reward, std_reward = evaluate_policy(
                model_to_use, 
                wrapper, 
                n_eval_episodes=eval_episodes, 
                deterministic=True
            )
            print(f"Agent {self.id} finished predicting.")
            print(f"Mean reward: {mean_reward} +/- {std_reward}")
            self.result_message = f"Mean reward: {mean_reward} +/- {std_reward}"
        except Exception as e:
            print(f"Error occurred during prediction: {e}")
            self.error_message = str(e)
        finally:
            # clean up
            if wrapper:
                wrapper.close()
            self.update_status("idle")

    def update_status(self, new_status: str):
        self.status = new_status
        self.status_timestamp = datetime.datetime.now()