import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional
import asyncio 

from agent import Agent 
import os

SIMULATOR_API_URL = os.getenv("SIMULATOR_API_URL")

app = FastAPI()

# global variables to hold the current simulator env and state
agents_list = {}

# This helper function wraps the blocking "train" call
async def run_training_in_background(agent: Agent, simulator_id: str, simulator_environment: str, api_url: str, total_timesteps: int = 20000, filename: str = None):
    """
    Wrapper to run the blocking train() in a separate thread
    so it doesn't block the main asyncio event loop.
    """
    try:
        agent.update_status("training")
        await asyncio.to_thread(
            agent.train, 
            simulator_id, 
            simulator_environment, 
            api_url,
            total_timesteps,
            filename
        )
    except Exception as e:
        print(f"Training task for agent {agent.id} failed: {e}")
        agent.update_status("idle")

# This helper function wraps the blocking "predict" call
async def run_prediction_in_background(agent: Agent, simulator_id: str, 
                                       simulator_environment: str, 
                                       api_url: str, 
                                       eval_episodes: int = 10, 
                                       save_filename: Optional[str] = None):
    """Wrapper to run the blocking predict() in a separate thread."""
    try:
        agent.update_status("predicting")
        await asyncio.to_thread(
            agent.predict,
            simulator_id,
            simulator_environment,
            api_url,
            eval_episodes,
            save_filename
        )
    except Exception as e:
        print(f"Prediction task for agent {agent.id} failed: {e}")
        agent.update_status("idle")


@app.post("/agents")
async def create_agent():
    agent = Agent()
    agents_list[agent.id] = agent
    return {"message": "Created agent", "agent_id": agent.id}

@app.get("/agents")
async def list_agents():
    return {"agents": list(agents_list.keys())}

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = agents_list.get(agent_id)
    if agent:
        return {"agent_id": agent.id, "status": agent.status,"elapsed_time": str(datetime.datetime.now() - agent.status_timestamp),"result_message": agent.result_message, "error_message": agent.error_message}
    else:
        raise HTTPException(status_code=404, detail="Agent not found")
    
@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    global agents_list
    agent = agents_list.get(agent_id)
    if agent:
        if agent.status != "idle":
            raise HTTPException(status_code=400, detail="Cannot delete an agent that is busy. Please wait until it is idle.")
        
        del agents_list[agent_id]
        return {"message": f"Deleted agent {agent_id}"}
    else:
        raise HTTPException(status_code=404, detail="Agent not found")

class TrainRequest(BaseModel):
    agent_id: str
    simulator_id: str
    simulator_environment: str
    api_url: Optional[str] = SIMULATOR_API_URL
    total_timesteps: Optional[int] = 20000
    filename: Optional[str] = None

@app.post("/agents/{agent_id}/train")
async def train_agent(agent_id: str, request_body: TrainRequest): # <-- Use the model
    agent = agents_list.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent.status != "idle":
        raise HTTPException(status_code=400, detail="Agent is currently busy. Please wait until it is idle.")
    
    asyncio.create_task(run_training_in_background(
        agent, 
        request_body.simulator_id, 
        request_body.simulator_environment, 
        request_body.api_url,
        request_body.total_timesteps,
        request_body.filename
    ))
    return {"message": f"Agent {agent_id} training started."}


class PredictRequest(BaseModel):
    agent_id: str
    simulator_id: str
    simulator_environment: str
    api_url: Optional[str] = SIMULATOR_API_URL
    eval_episodes: Optional[int] = 10
    save_filename: Optional[str] = None

@app.post("/agents/{agent_id}/predict")
async def predict_agent(agent_id: str, request_body: PredictRequest): # <-- Use the model
    agent = agents_list.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.status != "idle":
        raise HTTPException(status_code=400, detail="Agent is currently busy. Please wait until it is idle.")
    
    asyncio.create_task(run_prediction_in_background(
        agent, 
        request_body.simulator_id, 
        request_body.simulator_environment, 
        request_body.api_url, 
        request_body.eval_episodes, 
        request_body.save_filename
    ))
    
    return {"message": f"Agent {agent_id} prediction started."}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8081)