# SMARTS — Synchronous Multi‑Agent Reinforcement Learning Transit Simulator

SMARTS (Synchronous Multi‑Agent Reinforcement‑Learning Transit Simulator) is a research‑oriented simulation framework designed to study operational decision making in public transportation systems. The simulator models the core components of a transit system including vehicles, passengers, stops, routes, and network topology. It provides a structured environment for testing scheduling policies, dispatch strategies, and reinforcement learning based control algorithms.

The framework was developed to support experiments in adaptive transit operations where decisions must be made dynamically in response to passenger demand, vehicle locations, and network conditions. SMARTS enables researchers to prototype algorithms and evaluate their performance in a controlled simulation environment.

---

# Overview

Public transit systems face complex operational challenges such as irregular passenger demand, vehicle bunching, and inefficient scheduling. Traditional optimization and heuristic approaches often assume static conditions and struggle to adapt in real time. Reinforcement learning offers a promising alternative because agents can learn policies through interaction with the environment.

SMARTS provides a simulation environment where multiple agents interact synchronously with a transit network. Each agent observes the state of the system and decides actions such as dispatching vehicles, adjusting service frequency, or responding to demand patterns. The simulator advances in discrete time steps and updates all entities simultaneously, ensuring consistent system dynamics.

The framework is intentionally lightweight and modular so that researchers can easily modify system components or integrate new algorithms.

---

# Features

## Transit System Modeling

SMARTS explicitly models the key components of a public transit system:

* **Stops (Nodes)** represent passenger boarding and alighting locations.
* **Routes** define ordered sequences of stops served by transit vehicles.
* **Vehicles (Buses)** travel along routes and transport passengers.
* **Passengers** are generated according to configurable demand processes.
* **Network Topology** represents connectivity between stops and routes.

These components interact dynamically during the simulation to replicate the operational behavior of a real transit system.

## Multi‑Agent Reinforcement Learning Environment

The simulator is designed for reinforcement learning research:

* Multiple agents can operate simultaneously.
* Each agent receives system observations.
* Agents choose actions that affect the system state.
* Rewards can be defined based on performance metrics such as passenger wait time or service reliability.

This design allows the study of decentralized or coordinated control strategies.

## Synchronous Simulation Engine

SMARTS uses a **synchronous step‑based simulation** where:

1. The environment collects observations.
2. Agents select actions.
3. The environment updates all system entities.
4. Rewards and new observations are produced.

This structure closely mirrors reinforcement learning environments used in modern ML frameworks.

## Configurable Experiments

Simulation experiments are controlled through a JSON configuration file. Researchers can easily modify:

* network structure
* vehicle fleet size
* passenger generation parameters
* simulation time horizon
* decision intervals

This allows rapid experimentation without changing the source code.

## Modular Codebase

SMARTS is intentionally modular. Each transit component is implemented in a separate module so that researchers can extend or replace functionality without affecting the entire system.

---

# Repository Structure

```
├── agent.py                # Agent decision logic and policies
├── bus.py                  # Vehicle (bus) state and movement behavior
├── env.py                  # Simulation environment and interaction loop
├── functions.py            # Utility/helper functions
├── logger.py               # Logging and experiment output utilities
├── node.py                 # Stop/node representation
├── passenger.py            # Passenger generation and travel behavior
├── route.py                # Route definitions and stop sequences
├── topology.py             # Transit network topology representation
├── transit_system.py       # Main simulation runner and system manager
├── transit_system_config.json  # Simulation configuration file
├── LICENSE
└── README.md
```

Each module encapsulates a specific component of the transit system. This separation simplifies debugging, experimentation, and future development.

---

# System Architecture

SMARTS follows a layered architecture:

**Transit Network Layer**

* Nodes (stops)
* Routes
* Network topology

<img width="685" height="512" alt="image" src="https://github.com/user-attachments/assets/81576ee7-da3e-4156-a221-2f1f3c4fd2b5" />

**Operational Layer**

* Vehicles
* Passenger flows
* Route traversal

**Control Layer**

* Agents
* Decision policies
* Reinforcement learning models

  <img width="685" height="512" alt="image" src="https://github.com/user-attachments/assets/2c65da63-babd-42fc-b0ce-f293b78cb978" />

**Simulation Layer**

* Environment
* Time‑step updates
* Reward and observation generation

<!-- <ul>
  <li></li>
  <li></li>
  
</ul> -->



---

# Simulation Workflow

A typical simulation proceeds as follows:

1. The transit network is initialized from the configuration file.
2. Vehicles are deployed on their respective routes.
3. Passenger demand is generated at stops.
4. The environment produces observations for agents.
5. Agents select actions (e.g., dispatch or hold decisions).
6. The environment advances one time step.
7. System states and metrics are updated.
8. The process repeats until the simulation horizon is reached.

This workflow allows agents to continuously interact with the environment and learn operational policies.

---

# Installation

Clone the repository:

```
git clone <repository_url>
cd SMARTS
```

Install dependencies:

```
pip install -r requirements.txt
```

Recommended dependencies:

* numpy
* pandas
* networkx
* matplotlib
* torch
* torch_geometric

---

# Configuration

Simulation experiments are defined in the file:

```
transit_system_config.json
```

Key configuration parameters include:

**Network configuration**

* stop locations
* route definitions
* network connectivity

**Vehicle configuration**

* fleet size
* vehicle capacity
* operating speed

**Passenger demand**

* arrival rates
* origin–destination patterns

**Simulation parameters**

* time step length
* simulation duration
* decision interval

The configuration file allows rapid experimentation without modifying code.

---

# Running the Simulator

Experiments are typically executed using the provided notebook:

```
test.ipynb
```

The notebook initializes the transit system, runs the simulation loop, and records system metrics.

Users can modify the notebook to integrate reinforcement learning algorithms, training loops, or visualization routines.

---

# Outputs and Metrics

During simulation the system records operational metrics such as:

* passenger waiting time
* vehicle occupancy
* service frequency
* travel time
* system throughput

These metrics can be used to evaluate control strategies or compare different policies.

Generated logs and outputs are saved through the logging module.

---

# Visualization

SMARTS supports visual inspection of simulation results. Typical visualizations include:

* transit network diagrams
* vehicle trajectories
* passenger demand patterns
* simulation snapshots

Example figures of the simulation environment and network structure can be added below.

---

# Example Experiments

SMARTS can be used to explore a variety of research questions:

* Evaluating fleet size impacts on passenger waiting time
* Studying vehicle bunching and headway instability
* Testing dynamic dispatch strategies
* Comparing rule‑based control with reinforcement learning
* Investigating adaptive frequency setting

---

# Extending SMARTS

Researchers can extend the simulator in several ways:

* Implement new RL agents in `agent.py`
* Add alternative passenger demand models
* Integrate graph neural networks for network‑aware control
* Implement additional performance metrics

Because the codebase is modular, new components can be integrated with minimal changes to existing modules.

---

# Contributing

Contributions are welcome.

If you would like to improve SMARTS:

1. Fork the repository
2. Create a new feature branch
3. Implement your changes
4. Submit a pull request

Please keep contributions focused and document new functionality.

---

# License

This project is distributed under the terms of the LICENSE file included in the repository.

---

# Acknowledgement

SMARTS was developed as part of research on reinforcement learning applications for public transportation systems. The simulator aims to provide a flexible experimentation platform for researchers studying adaptive and intelligent transit operations.

---
