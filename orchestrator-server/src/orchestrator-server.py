from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional
import httpx
import os

SIMULATOR_API_URL = os.getenv("SIMULATOR_API_URL")
AGENT_API_URL = os.getenv("AGENT_API_URL")

app = FastAPI()

@app.post("/agents")
async def create_agent(agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.post("/agents")
    return response.json()


class SimulatorConfig(BaseModel):
    environment: str                        # name of the environment
    config: Optional[Dict[str, Any]] = None # optional config parameters

@app.post("/simulators")
async def create_simulator(simulator_config: SimulatorConfig, simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.post("/simulators", json=simulator_config.model_dump())
    return response.json()


@app.get("/simulators")
async def list_simulators(simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.get("/simulators")
    return response.json()

@app.get("/agents")
async def list_agents(agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.get("/agents")
    return response.json()

@app.get("/simulators/{simulator_id}")
async def get_simulator(simulator_id: str, simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.get(f"/simulators/{simulator_id}")
    return response.json()

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.get(f"/agents/{agent_id}")
    return response.json()

@app.delete("/simulators/{simulator_id}")
async def delete_simulator(simulator_id: str, simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.delete(f"/simulators/{simulator_id}")
    return response.json()

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.delete(f"/agents/{agent_id}")
    return response.json()

class TrainRequest(BaseModel):
    agent_id: str
    simulator_id: str
    simulator_environment: str
    api_url: Optional[str] = SIMULATOR_API_URL
    total_timesteps: Optional[int] = 20000
    filename: Optional[str] = None

@app.post("/services/train")
async def train_agent(request: TrainRequest):
    # send the request to the agent server
    async with httpx.AsyncClient(base_url=AGENT_API_URL) as client:
        response = await client.post(f"/agents/{request.agent_id}/train", json=request.model_dump())
    return response.json()

class PredictRequest(BaseModel):
    agent_id: str
    simulator_id: str
    simulator_environment: str
    api_url: Optional[str] = SIMULATOR_API_URL
    eval_episodes: Optional[int] = 10
    save_filename: Optional[str] = None

@app.post("/services/predict")
async def predict_agent(request: PredictRequest):
    # send the request to the agent server
    async with httpx.AsyncClient(base_url=AGENT_API_URL) as client:
        response = await client.post(f"/agents/{request.agent_id}/predict", json=request.model_dump())
    return response.json()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8082)