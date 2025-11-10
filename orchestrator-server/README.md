# Orchestrator Server API Reference

Welcome to the Orchestrator Server API! This API allows you to manage reinforcement learning **agents** and **simulation environments**. You can use these endpoints to create, configure, and delete resources, and then use the service endpoints to **train** agents within simulators or **run predictions** using trained models.

> **Note:** All endpoint paths are relative to your server's base URL (e.g., `http://localhost:8082`).

---

## Agent Management

Endpoints for creating, viewing, and deleting agents.

### `POST /agents`
* **Description:** Creates a new agent with a given configuration. The response will contain the uuid of the created agent if success.

### `GET /agents`
* **Description:** Retrieves a list of all available agents.

### `GET /agents/{agent_id}`
* **Description:** Retrieves detailed information about a specific agent.

### `DELETE /agents/{agent_id}`
* **Description:** Deletes a specific agent and detaches it from any simulation environment.

---

## Simulator Management

Endpoints for creating, viewing, and deleting simulation environments.

### `POST /simulators`
* **Description:** Creates a new simulation environment.
* **Request Body (JSON):**
    ```json
    {
      "environment": "LunarLander-v3",
      "config": {
        "continous": false,
        "gravity": -10.0,
        "enable_wind": false,
        "wind_power": 15.0,
        "turbulence_power": 1.5
      }
    }
    ```
    * `environment` (string, required): The name of the environment (e.g., from Gymnasium).
    * `config` (object, optional): Environment-specific configuration options.

### `GET /simulators`
* **Description:** Retrieves a list of all active simulators.

### `GET /simulators/{simulator_id}`
* **Description:** Retrieves detailed information about a specific simulator.

### `DELETE /simulators/{simulator_id}`
* **Description:** Deletes a specific simulation environment.

---

## Services

Endpoints for triggering core processes like training and prediction.

### `POST /services/train`
* **Description:** Starts a training process, linking a specific agent with a simulator.
* **Request Body (JSON):**
    ```json
    {
      "agent_id": "your-agent-uuid",
      "simulator_id": "your-simulator-uuid",
      "simulator_environment": "LunarLander-v3",
      "api_url": "http://simulator-server:8080",
      "total_timesteps": 20000,
      "filename": "my-trained-model"
    }
    ```
    * `simulator_environment`: currently only supports "LunarLander-v3"
    * `api_url` (string, optional): Default is `http://simulator-server:8080`.
    * `total_timesteps` (int, optional): Default is `20000`.
    * `filename` (string, optional): Name to save the trained model as.

### `POST /services/predict`
* **Description:** Starts an evaluation (prediction) process, running a trained agent in a simulator.
* **Request Body (JSON):**
    ```json
    {
      "agent_id": "your-agent-uuid",
      "simulator_id": "your-simulator-uuid",
      "simulator_environment": "LunarLander-v3",
      "api_url": "http://simulator-server:8080",
      "eval_episodes": 10,
      "save_filename": "my-prediction-results"
    }
    ```
    * `api_url` (string, optional): Default is `http://simulator-server:8080`.
    * `eval_episodes` (int, optional): Default is `10`.
    * `save_filename` (string, optional): Name to save prediction results/videos.

---

## Health Check

### `GET /health`
* **Description:** Checks the health and availability of the API server.
* **Success Response (JSON):**
    ```json
    {
      "status": "ok"
    }
    ```

---

## Typical Workflow

Here is a typical sequence of API calls for a complete task:

1.  **Create a Simulator:** Send a `POST /simulators` request with your environment details. Note the `simulator_id` from the response.
2.  **Create an Agent:** Send a `POST /agents` request with your agent configuration. Note the `agent_id` from the response.
3.  **Start Training:** Send a `POST /services/train` request using the `agent_id` and `simulator_id` you just received.
4.  **Monitor (Optional):** You can poll `GET /agents/{agent_id}` or `GET /simulators/{simulator_id}` to check their status (e.g., "training", "idle").
5.  **Run Prediction:** Once training is complete, send a `POST /services/predict` request (using the same IDs) to evaluate your agent's performance.
6.  **Clean Up:** When finished, you can delete the resources using `DELETE /agents/{agent_id}` and `DELETE /simulators/{simulator_id}`.