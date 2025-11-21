import pandas as pd
import time
from Neural_networks import *
import matplotlib.pyplot as plt
from tqdm import tqdm

############## Helper functions for comparison #######################################################################

def comparing_methods_initialization_after_pruning(amount_of_repeats, rounds, method_1, method_2):
    beginning = time.time() #XXx remove
    df_accuracies_method_1 = pd.DataFrame()
    df_accuracies_method_2 = pd.DataFrame()
    
    for i in tqdm(range(amount_of_repeats)):
        print(f"\nIteration for average : {i + 1}/{amount_of_repeats}")
        #All the iterations for the first method
        df_acc_method_1,_ = method_1()
        for j in range (len(df_acc_method_1)):
            df_acc_method_1[j]['Iteration'] = i + 1
        df_accuracies_method_1 = pd.concat([df_accuracies_method_1, pd.DataFrame(df_acc_method_1)])

        #All the iterations for the LTH method
        df_acc_method_2,_ = method_2()
        for j in range (len(df_acc_method_2)):
            df_acc_method_2[j]['Iteration'] = i + 1
        df_accuracies_method_2 = pd.concat([df_accuracies_method_2, pd.DataFrame(df_acc_method_2)])

    df_avg_accuracies_method_1 = []
    df_avg_accuracies_method_2 = []
    for pruning_round in range (rounds): #xxx - find rounds in another way. 
        for df, df_avg in [(df_accuracies_method_1, df_avg_accuracies_method_1), (df_accuracies_method_2, df_avg_accuracies_method_2)]:
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

    df_avg_accuracies_method_1 = pd.DataFrame(df_avg_accuracies_method_1)
    df_avg_accuracies_method_2 = pd.DataFrame(df_avg_accuracies_method_2)

    print((time.time()- beginning)/60)#XXx remove
    
    return df_avg_accuracies_method_1, df_avg_accuracies_method_2



def comparing_methods_plotting(df_avg_accuracies_method_1, df_avg_accuracies_method_2, method_1_name, method_2_name):

    plt.errorbar(df_avg_accuracies_method_1['Pruning Percentage'],
                df_avg_accuracies_method_1['Avg Test Accuracy'],
                yerr=[df_avg_accuracies_method_1['Avg Test Accuracy'] - df_avg_accuracies_method_1['Min Test Accuracy'],
                        df_avg_accuracies_method_1['Max Test Accuracy'] - df_avg_accuracies_method_1['Avg Test Accuracy']],
                label = method_1_name)
    plt.errorbar(df_avg_accuracies_method_2['Pruning Percentage'],
                df_avg_accuracies_method_2['Avg Test Accuracy'],
                yerr=[df_avg_accuracies_method_2['Avg Test Accuracy'] - df_avg_accuracies_method_2['Min Test Accuracy'],
                        df_avg_accuracies_method_2['Max Test Accuracy'] - df_avg_accuracies_method_2['Avg Test Accuracy']],
                label = method_2_name)
    
    plt.xlabel("Pruning Percentage")
    plt.ylabel("Test Accuracy")
    plt.title(f"Comparing the initialization methods : {method_1_name} vs {method_2_name}") #xxx Change title 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/comparing_initialization_post_pruning_{method_1_name.replace(' ','')}_vs_{method_2_name.replace(' ','')}.png") #xxx Change file title 





############## Comparing the two strategies in the appendix ##########################################################

"""
In the Lottery Ticket Hypothesis paper, the authors propose two different iterative pruning strategies to find the winning tickets. 

Strategy 1: Iterative pruning with resetting.
1. Randomly initialize a neural network f(x; m⊙θ) where θ= θ_0 and m= 1^(|θ|) is a mask.
2. Train the network for jiterations, reaching parameters m⊙θ_j.
3. Prune s% of the parameters, creating an updated mask m' where Pm' = (Pm - s)%.
4. Reset the weights of the remaining portion of the network to their values in θ0. That is, let
θ= θ_0.
5. Let m = m' and repeat steps 2 through 4 until a sufficiently pruned network has been
obtained.


Strategy 2: Iterative pruning with continued training.
1. Randomly initialize a neural network f(x; m⊙θ) where θ= θ_0 and m= 1^(|θ|) is a mask.
2. Train the network for jiterations.
3. Prune s% of the parameters, creating an updated mask m' where Pm' = (Pm - s)%.
4. Let m= m' and repeat steps 2 and 3 until a sufficiently pruned network has been obtained.
5. Reset the weights of the remaining portion of the network to their values in θ_0. That is, let
θ= θ_0.

The difference between the two strategies is the reinitialization of the weights after pruning. 
At step 3, s% of the parameters are pruned, and we create a mask. In strategy 1, the unmasked 
weights are being reset to their value from θ_0 and then the pruning steps are being repeated.
In strategy 2, after pruning, the weights are kept at their post-trained values to then be 
trained again with only the unpruned weights. It is at the end when the model has been sufficiently 
pruned that we train it one final time with θ_0 .

According to the paper, the first strategy offers better results. In this section, we propose to 
check this claim.

"""
total_prune_percent = 99
rounds = 7
amount_of_repeats = 5

def method_1_LTH_strategy1():
    return iterative_pruning_MNIST(total_prune_percent=total_prune_percent, rounds=rounds, LTH=True)

def method_2_LTH_strategy2():
    return iterative_pruning_MNIST(total_prune_percent=total_prune_percent, rounds=rounds, strategy_1=False)

df_avg_acc_method_1, df_avg_acc_method_2 = comparing_methods_initialization_after_pruning(amount_of_repeats = amount_of_repeats, rounds = rounds, method_1=method_1_LTH_strategy1, method_2=method_2_LTH_strategy2)
comparing_methods_plotting(df_avg_acc_method_1, df_avg_acc_method_2, "Strategy 1", "Strategy 2")



############## Comparing initialization after pruning ##########################################################

"""
On page 2 of the Lottery Ticket Hypothesis paper, the authors claim that the strength of their unique pruning technique
comes from the fact that after pruning, the weights are set back to their original weights (from the dense model). From
there, the model is retrained and the test accuracies are as good, if not better then they were originally.

We want to test this theory by randomly reinitializing the not-pruned weights, and seeing the test accuracies.
"""
total_prune_percent = 99
rounds = 7
amount_of_repeats = 5

def method_1_LTH():
    return iterative_pruning_MNIST(total_prune_percent=total_prune_percent, rounds=rounds, LTH=True)

def method_2_random():
    return iterative_pruning_MNIST(total_prune_percent=total_prune_percent, rounds=rounds, LTH=False)

df_avg_acc_method_1, df_avg_acc_method_2 = comparing_methods_initialization_after_pruning(amount_of_repeats = amount_of_repeats, rounds = rounds, method_1=method_1_LTH, method_2=method_2_random)
comparing_methods_plotting(df_avg_acc_method_1, df_avg_acc_method_2, "LTH", "Random Initialisation")









