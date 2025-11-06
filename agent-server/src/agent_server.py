from fastapi import FastAPI, HTTPException
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
async def run_training_in_background(agent: Agent, simulator_id: str, simulator_environment: str, api_url: str):
    """
    Wrapper to run the blocking train() in a separate thread
    so it doesn't block the main asyncio event loop.
    """
    try:
        # This is the key:
        # It runs the normal (sync) agent.train() function in a thread pool.
        # The 'await' here only blocks *this background task*, not the main server.
        await asyncio.to_thread(
            agent.train, 
            simulator_id, 
            simulator_environment, 
            api_url
        )
    except Exception as e:
        print(f"Training task for agent {agent.id} failed: {e}")
        # agent.train() has a finally block, but we'll set it here just in case
        agent.status = "idle"

# This helper function wraps the blocking "predict" call
async def run_prediction_in_background(agent: Agent, simulator_id: str, 
                                       simulator_environment: str, 
                                       api_url: str, 
                                       eval_episodes: int = 100, 
                                       load_path: Optional[str] = None):
    """Wrapper to run the blocking predict() in a separate thread."""
    try:
        await asyncio.to_thread(
            agent.predict,
            simulator_id,
            simulator_environment,
            api_url,
            eval_episodes,
            load_path
        )
    except Exception as e:
        print(f"Prediction task for agent {agent.id} failed: {e}")
        agent.status = "idle"


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
        return {"agent_id": agent.id, "status": agent.status}
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

@app.post("/agents/{agent_id}/train")
async def train_agent(agent_id: str, simulator_id: str, simulator_environment: str, api_url: str = SIMULATOR_API_URL):
    agent = agents_list.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent.status != "idle":
        raise HTTPException(status_code=400, detail="Agent is currently busy. Please wait until it is idle.")
    
    # 1. Set status immediately
    agent.status = "training" 
    # 2. Schedule the background task and return
    asyncio.create_task(run_training_in_background(
        agent, simulator_id, simulator_environment, api_url
    ))
    
    return {"message": f"Agent {agent_id} training started."}


@app.post("/agents/{agent_id}/predict")
async def predict_agent(agent_id: str, simulator_id: str, simulator_environment: str, api_url: str = SIMULATOR_API_URL, eval_episodes: Optional[int] = 100, load_path: Optional[str] = None):
    agent = agents_list.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent.status != "idle":
        raise HTTPException(status_code=400, detail="Agent is currently busy. Please wait until it is idle.")

    agent.status = "predicting"
    asyncio.create_task(run_prediction_in_background(
        agent, simulator_id, simulator_environment, api_url, eval_episodes, load_path
    ))
    
    return {"message": f"Agent {agent_id} prediction started."}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8081)