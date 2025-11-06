from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Dict, Any, Optional
from pydantic import BaseModel

from simulator import LunarLanderSimulator

# Server to manage simulator envs
app = FastAPI()

# global variables to hold the current simulator env and state
simulator_list = {}


class SimulatorConfig(BaseModel):
    environment: str                        # name of the environment
    config: Optional[Dict[str, Any]] = None # optional config parameters

@app.post("/simulators")
async def create_simulator(config: SimulatorConfig):
    global simulator_list

    if config.environment == "LunarLander-v3":   
        config_dict = config.config if config.config is not None else {}
        simulator = LunarLanderSimulator(
            config_dict.get("continuous", False), 
            config_dict.get("gravity", -10.0),
            config_dict.get("enable_wind", False), 
            config_dict.get("wind_power", 15.0),
            config_dict.get("turbulence_power", 1.5)
        )
        simulator_list[simulator.id] = simulator
        return {"message": "Created simulator", "simulator_id": simulator.id}
    else:
        raise HTTPException(status_code=400, detail="Invalid environment")


@app.get("/simulators")
async def list_simulators():
    return {"simulators": list(simulator_list.keys())}


@app.get("/simulators/{simulator_id}")
async def get_simulator(simulator_id: str):
    simulator = simulator_list.get(simulator_id)
    if simulator:
        return simulator.to_json()
    else:
        raise HTTPException(status_code=404, detail="Simulator not found")


@app.delete("/simulators/{simulator_id}")
async def delete_simulator(simulator_id: str):
    global simulator_list
    simulator = simulator_list.get(simulator_id)
    if simulator:
        simulator.close()
        del simulator_list[simulator_id]
        return {"message": f"Deleted simulator {simulator_id}"}
    else:
        raise HTTPException(status_code=404, detail="Simulator not found")


@app.post("/simulators/{simulator_id}/reset")
async def reset_simulator(simulator_id: str):
    simulator = simulator_list.get(simulator_id)
    if simulator:
        state, info = simulator.reset()
        return {
            "state": state.tolist(),
            "info": info
        }
    else:
        raise HTTPException(status_code=404, detail="Simulator not found")


class StepAction(BaseModel):
    action: int

@app.post("/simulators/{simulator_id}/step")
async def step_simulator(simulator_id: str, step_action: StepAction):
    simulator = simulator_list.get(simulator_id)
    if simulator:
        state, reward, terminated, truncated, info = simulator.step(step_action.action)
        return {
            "state": state.tolist(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    else:
        raise HTTPException(status_code=404, detail="Simulator not found")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # run the app, listen to the port
    print("Starting simulator server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)