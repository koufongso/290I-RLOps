## Simulator Server API

### Endpoints

* **POST** `/simulators`
    * Creates a new simulation environment with a given configuration.

* **GET** `/simulators`
    * Get the information about the simulation environment collection.

* **GET** `/simulators/{simulator_id}`
    * Get the information about the specific simulation environment.

* **DELETE** `/simulators/{simulator_id}`
    * Deletes/closes the specified simulation environment.

* **POST** `/simulators/{simulator_id}/reset`
    * Resets the specified simulation environment to its initial state.

* **POST** `/simulators/{simulator_id}/step`
    * Performs one simulation step using the provided action.

* **GET** `/health`
    * Checks the health status of the API server.


