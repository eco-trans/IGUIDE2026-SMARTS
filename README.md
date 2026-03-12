# SMARTS — Synchronous Multi Agent Reinforcement-Learning Transit Simulator

SMARTS is a lightweight simulation framework for public transit research. It models vehicles, passengers, stops, routes, and a network topology, enabling experiments with control policies, scheduling strategies, and reinforcement learning agents.

---

## Features

* Complete transit network simulation (nodes, routes, vehicles, passengers).
* Synchronous multi-agent step environment.
* Pluggable agent logic for RL or heuristic policies.
* JSON-based configuration for quick experiment setup.
* Simple, readable codebase designed for research and extension.

---

## Repository Structure

```
├── agent.py                # Agent decision logic
├── bus.py                  # Vehicle (bus) model
├── env.py                  # Simulation environment
├── functions.py            # Utility functions
├── logger.py               # Logging utilities
├── node.py                 # Stop/node representation
├── passenger.py            # Passenger behavior model
├── route.py                # Route and stop sequences
├── topology.py             # Network topology tools
├── transit_system.py       # Main simulation runner
├── transit_system_config.json  # Config file
├── LICENSE
└── README.md
```

---

## Requirements

Create a `requirements.txt` file and install dependencies with:

```
pip install -r requirements.txt
```

Recommended dependencies:

* numpy
* pandas
* networkx
* matplotlib
* pytorch
* pytorch geometric

---

## Quick Start

### 1. Edit Configuration

Update `transit_system_config.json` to define:

* routes and stop sequences
* vehicles and capacities
* passenger generation parameters
* simulation horizon and step size
* agent parameters

### 2. Run the Simulator

```
python transit_system.py --config transit_system_config.json
```

### 3. Analyze Outputs

Generated logs and metrics can be found in the output directory defined in the config.

---

## Typical Use Cases

* Evaluate fleet size and its impact on waiting time.
* Compare rule-based vs. RL-based scheduling.
* Test headway control strategies.
* Explore demand-responsive or event-triggered dispatch logic.

---

## Contributing

Pull requests and issues are welcome.
Please keep contributions focused and include brief examples when adding new features.

---

## License

Distributed under the terms of the LICENSE file included in this repository.
