import pandas as pd
import numpy as np
import time
from Neural_network import *
import matplotlib.pyplot as plt

############## Comparing initialization after pruning ##########################################################
"""
On page 2 of the Lottery Ticket Hypothesis paper, the authors claim that the strength of their unique pruning technique
comes from the fact that after pruning, the weights are set back to their original weights (from the dense model). From
there, the model is retrained and the test accuracies are as good, if not better then they were originally.

We want to test this theory by randomly reinitializing the not-pruned weights, and seeing the test accuracies.
"""

def comparing_initialization_after_pruning(amount_of_repeats, total_prune_percent=99, rounds=7):
    beginning = time.time() #XXx remove
    df_accuracies_LTH = pd.DataFrame()
    df_accuracies_not_LTH = pd.DataFrame()

    for i in range (amount_of_repeats):
        print(f"Iteration for average : {i + 1}/{amount_of_repeats}")

        #All the iterations for the LTH method
        df_acc_LTH,_ = iterative_pruning(total_prune_percent=total_prune_percent, rounds=rounds)
        for j in range (len(df_acc_LTH)):
            df_acc_LTH[j]['Iteration'] = i + 1
        df_accuracies_LTH = pd.concat([df_accuracies_LTH, pd.DataFrame(df_acc_LTH)])

        #All the iterations for the LTH method
        df_acc_not_LTH,_ = iterative_pruning(total_prune_percent=total_prune_percent, rounds=rounds)
        for j in range (len(df_acc_not_LTH)):
            df_acc_not_LTH[j]['Iteration'] = i + 1
        df_accuracies_not_LTH = pd.concat([df_accuracies_not_LTH, pd.DataFrame(df_acc_not_LTH)])

    df_avg_accuracies_LTH = []
    df_avg_accuracies_not_LTH = []
    for pruning_round in range (rounds):
        for df, df_avg in [(df_accuracies_LTH, df_avg_accuracies_LTH), (df_accuracies_not_LTH, df_avg_accuracies_not_LTH)]:
            df_round = df[df['Round']=='Round '+str(pruning_round)]
            round_avg_of_test_acc = df_round['Test Accuracy (with training)'].mean()
            min_test_acc = df_round['Test Accuracy (with training)'].min()
            max_test_acc = df_round['Test Accuracy (with training)'].max()
            pruning_perc = df_round['Pruning percentage'].mean()

            round_avg_info = {'Round' : 'Round '+str(pruning_round), 
                            'Pruning Percentage' : pruning_perc,
                            'Avg Test Accuracy' : round_avg_of_test_acc,
                            'Min Test Accuracy' : min_test_acc,
                            'Max Test Accuracy' : max_test_acc}
            df_avg.append(round_avg_info)
    
    df_avg_accuracies_LTH = pd.DataFrame(df_avg_accuracies_LTH)
    df_avg_accuracies_not_LTH = pd.DataFrame(df_avg_accuracies_not_LTH)

    plt.plot(df_avg_accuracies_LTH['Pruning Percentage'], df_avg_accuracies_LTH['Avg Test Accuracy'], label = 'LTH')
    plt.plot(df_avg_accuracies_not_LTH['Pruning Percentage'], df_avg_accuracies_not_LTH['Avg Test Accuracy'], label = 'Random initialization')
    plt.xlabel("Pruning Percentage")
    plt.ylabel("Test Accuracy")
    plt.title("Comparing the initialization post pruning.")
    plt.legend()
    plt.savefig("plots/randomly_initialized.png")

    print((time.time()- beginning)/60)#XXx remove


comparing_initialization_after_pruning(amount_of_repeats = 5, rounds = 7)











