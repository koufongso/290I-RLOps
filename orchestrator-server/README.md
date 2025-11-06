## Orchestrator Server API

### Endpoints

* **POST** `/agents`
    * Creates a new agent with a given configuration.

* **GET** `/agents`
    * Get the information about the agent collection.

* **POST** `/simulators`
    * Creates a new agent with a given configuration.

* **GET** `/simulators`
    * Creates a new agent with a given configuration.

* **GET** `/agents/{agent_id}`
    * Get the information about the specific agent.

* **GET** `/simulators/{simulator_id}`
    * Get the information about the specific agent.

* **DELETE** `/agents/{agent_id}`
    * Dettached the agent from the simulation environemnt and delete this agent

* **DELETE** `/simulators/{simulator_id}`
    * Dettached the agent from the simulation environemnt and delete this agent

* **POST** `/services/train`
    * Start training process with a request simulator, agent id pair

* **POST** `/services/predict`
    * Start prediction process with a request simulator, agent id pair

* **GET** `/health`
    * Checks the health status of the API server.


