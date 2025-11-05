## Agent Server API

### Endpoints

* **POST** `/agents`
    * Creates a new agent with a given configuration.

* **GET** `/agents`
    * Get the information about the agent collection.

* **GET** `/agents/{agent_id}`
    * Get the information about the specific agent.

* **DELETE** `/agents/{agent_id}`
    * Dettached the agent from the simulation environemnt and delete this agent

* **POST** `/agents/{agent_id}/train`
    * Start training process with a request simulation environment

* **POST** `/agents/{agent_id}/predict`
    * Start prediction process with a request simulation environment

* **GET** `/health`
    * Checks the health status of the API server.


