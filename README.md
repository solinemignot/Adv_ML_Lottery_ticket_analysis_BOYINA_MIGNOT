# An Empirical Study of the Lottery Ticket Hypothesis
Authors: Leena BOYINA and Soline MIGNOT.

This repository contains the codebase for a project investigating the Lottery Ticket Hypothesis (LTH). The work reproduces key experiments from Frankle & Carbin (2019) and extends them through comparisons with classical pruning strategies, late rewinding, and the Strong Lottery Ticket Hypothesis.

The project focuses on empirically studying the roles of sparsity, initialization, optimization dynamics, and network topology in over-parameterized neural networks.

## Folders:

### data/

Contains the datasets used in this project:
- MNIST (handwritten digits)
- CIFAR-10 (natural images)
Datasets are downloaded automatically if not already present.

### df_accuracies/

Stores all experimental results as Pandas DataFrames, including:
- test accuracy
- final training loss 
- execution time 
at each epoch.

These files are used to generate the plots and to support quantitative analysis.

### plots/

Contains all figures produced throughout the project (accuracy curves, loss curves, timing comparisons, strong LTH sweep, etc.), many of which are directly used in the final report.


## Notebooks

### code_gpu.ipynb

Main notebook containing:
- replication of the original Lottery Ticket Hypothesis experiments
- comparison between iterative pruning and one-shot pruning
- random reinitialization control experiments

This notebook handles the core training, pruning, evaluation, and data collection pipeline.

### code_extensions.ipynb
Extension-focused notebook containing:
- Strong Lottery Ticket Hypothesis experiments
- frozen-weight / learned-mask methodology
- parsity sweep analysis

This notebook is dedicated to investigating topology-driven learning beyond standard LTH.

## Python files

### **1. `Accessing_data.py`**
Handles data extraction, transformation, and loading (ETL).
- `load_mnist(batch_size)`: Downloads, normalizes, and creates iterators for the MNIST dataset (digits).
- `load_cifar(batch_size)`: Downloads, normalizes, and creates iterators for the CIFAR-10 dataset (images).

### **2. `Helper_functions.py`**
The algorithmic toolkit for training, evaluation, and pruning logic.
- `training_the_model(...)`: Standard PyTorch training loop (Forward pass $\rightarrow$ Loss $\rightarrow$ Backprop).
- `evaluate_the_model(...)`: Calculates the model's accuracy on test data without updating gradients.
- `get_weights(...)`: Saves a copy of the model's current parameters.
- `prune_by_magnitude(...)`: Identifies the smallest weights and creates a binary mask (0 for pruned, 1 for kept).
- `create_winning_ticket(...)`: **Key LTH function.** Resets the model parameters to a specific state ($\theta_0$ or $\theta_j$) and applies the mask.
- `randomly_reinitialize(...)`: Resets surviving weights to *new* random values (control experiment).
- `calculate_actual_prune_percent(...)` / `count_zeros(...)`: Utility functions to verify network sparsity.

### **3. `Neural_networks.py`**
*Defines architectures and orchestrates the Iterative Magnitude Pruning (IMP) algorithm.*
- `SimpleNN`: Defines a Multi-Layer Perceptron (MLP) architecture for MNIST.
- `SimpleCNN`: Defines a Convolutional Neural Network architecture for CIFAR-10.
- `dense_neural_network_...`: Trains the initial unpruned network to get the baseline $\theta_j$ and save $\theta_0$.
- `iterative_pruning_...`: **Core Logic.** Executes the loop: *Train $\rightarrow$ Prune $\rightarrow$ Reset Weights $\rightarrow$ Repeat*. Handles both "Winning Ticket" and "Standard Pruning" strategies.

### **4. `Comparing_results.py`**
The main execution script for running experiments and visualizing data.

* `comparing_methods_initialization_after_pruning(...)`: Runs multiple trials of two different methods (e.g., LTH vs. Random) to gather statistical averages.
* `comparing_methods_plotting(...)`: Generates graphs plotting Test Accuracy vs. Pruning Percentage with error bars.
* **Main execution blocks**: Configures and launches the comparisons defined in the project outline (Strategy 1 vs. 2, LTH vs. Random).

### **5. `Extension_analysis_functions.py`**
Implements the extension experiments beyond standard LTH.
This file contains the code for Late Rewinding and Strong Lottery Ticket Hypothesis experiments.

**Late Rewinding**: Implements training and iterative pruning on CIFAR-10 where weights are reset to an early checkpoint $\theta_k$ instead of the original initialization.

Key functions:
- dense_neural_network_CIFAR_Rewinding(...)
- iterative_pruning_CIFAR_Rewinding(...)

**Strong LTH (Edge-Popup)**:
Implements frozen-weight networks where only binary masks are learned. Introduces custom supermask layers and a straight-through estimator to optimize network topology.

Key components:
- SupermaskConv2d, SupermaskLinear, GetSubnet
- Conv2_Strong, run_strong_lth_experiment(...)
This module supports experiments analyzing training stability and topology-driven learning.
