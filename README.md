# Adv_ML_Lottery_ticket_analysis_BOYINA_MIGNOT
Implementation of the lottery ticket hypothesis for DNN and comparing its performance to other pruning techniques.

## Short description of the different .py files:


### **1. `Accessing_data.py`**
*Handles data extraction, transformation, and loading (ETL).*
* `load_mnist(batch_size)`: Downloads, normalizes, and creates iterators for the MNIST dataset (digits).
* `load_cifar(batch_size)`: Downloads, normalizes, and creates iterators for the CIFAR-10 dataset (images).

### **2. `Helper_functions.py`**
*The algorithmic toolkit for training, evaluation, and pruning logic.*
* `training_the_model(...)`: Standard PyTorch training loop (Forward pass $\rightarrow$ Loss $\rightarrow$ Backprop).
* `evaluate_the_model(...)`: Calculates the model's accuracy on test data without updating gradients.
* `get_weights(...)`: Saves a copy of the model's current parameters.
* `prune_by_magnitude(...)`: Identifies the smallest weights and creates a binary mask (0 for pruned, 1 for kept).
* `create_winning_ticket(...)`: **Key LTH function.** Resets the model parameters to a specific state ($\theta_0$ or $\theta_j$) and applies the mask.
* `randomly_reinitialize(...)`: Resets surviving weights to *new* random values (control experiment).
* `calculate_actual_prune_percent(...)` / `count_zeros(...)`: Utility functions to verify network sparsity.

### **3. `Neural_networks.py`**
*Defines architectures and orchestrates the Iterative Magnitude Pruning (IMP) algorithm.*
* `SimpleNN`: Defines a Multi-Layer Perceptron (MLP) architecture for MNIST.
* `SimpleCNN`: Defines a Convolutional Neural Network architecture for CIFAR-10.
* `dense_neural_network_...`: Trains the initial unpruned network to get the baseline $\theta_j$ and save $\theta_0$.
* `iterative_pruning_...`: **Core Logic.** Executes the loop: *Train $\rightarrow$ Prune $\rightarrow$ Reset Weights $\rightarrow$ Repeat*. Handles both "Winning Ticket" and "Standard Pruning" strategies.

### **4. `Comparing_results.py`**
*The main execution script for running experiments and visualizing data.*
* `comparing_methods_initialization_after_pruning(...)`: Runs multiple trials of two different methods (e.g., LTH vs. Random) to gather statistical averages.
* `comparing_methods_plotting(...)`: Generates graphs plotting Test Accuracy vs. Pruning Percentage with error bars.
* **Main execution blocks**: Configures and launches the comparisons defined in the project outline (Strategy 1 vs. 2, LTH vs. Random).
