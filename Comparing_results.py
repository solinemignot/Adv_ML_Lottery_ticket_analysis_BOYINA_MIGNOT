import pandas as pd
import numpy as np
from Neural_network import *
import matplotlib.pyplot as plt

############## Comparing initialization after pruning ##########################################################
"""
On page 2 of the Lottery Ticket Hypothesis paper, the authors claim that the strength of their unique pruning technique
comes from the fact that after pruning, the weights are set back to their original weights (from the dense model). From
there, the model is retrained and the test accuracies are as good, if not better then they were originally.

We want to test this theory by randomly reinitializing the not-pruned weights, and seeing the test accuracies.
"""

df_accuracies_LTH,_ = iterative_pruning(rounds=8)
df_accuracies_LTH = pd.DataFrame(df_accuracies_LTH)
print(df_accuracies_LTH)

df_accuracies_not_LTH,_ = iterative_pruning(rounds=8, LTH = False)
df_accuracies_not_LTH = pd.DataFrame(df_accuracies_not_LTH)
print(df_accuracies_not_LTH)

plt.plot(df_accuracies_LTH['Pruning percentage'], df_accuracies_LTH['Test Accuracy (with training)'], label = 'LTH')
plt.plot(df_accuracies_not_LTH['Pruning percentage'], df_accuracies_not_LTH['Test Accuracy (with training)'], label = 'not LTH')
plt.xlabel("Pruning percentage")
plt.ylabel("Test Accuracy")
plt.legend()
plt.savefig("plots/randomly_initialized.png")













