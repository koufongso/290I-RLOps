from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional
import httpx
import os

SIMULATOR_API_URL = os.getenv("SIMULATOR_API_URL")
AGENT_API_URL = os.getenv("AGENT_API_URL")


app = FastAPI()

# global variables to hold the agent-simulation experiments (id) pairs   
experiment_list = {}

@app.post("/agents")
async def create_agent(agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.post("/agents")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to create agent")
    return response.json()


class SimulatorConfig(BaseModel):
    environment: str                        # name of the environment
    config: Optional[Dict[str, Any]] = None # optional config parameters

@app.post("/simulators")
async def create_simulator(simulator_config: SimulatorConfig, simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.post("/simulators", json=simulator_config.model_dump())
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to create simulator")
    return response.json()


@app.get("/simulators")
async def list_simulators(simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.get("/simulators")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to list simulators")
    return response.json()

@app.get("/agents")
async def list_agents(agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.get("/agents")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to list agents")
    return response.json()

@app.get("/simulators/{simulator_id}")
async def get_simulator(simulator_id: str, simulator_api_url: str = SIMULATOR_API_URL):
    async with httpx.AsyncClient(base_url=simulator_api_url) as client:
        response = await client.get(f"/simulators/{simulator_id}")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get simulator")
    return response.json()

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, agent_api_url: str = AGENT_API_URL):
    async with httpx.AsyncClient(base_url=agent_api_url) as client:
        response = await client.get(f"/agents/{agent_id}")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get agent")
    return response.json()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8082)