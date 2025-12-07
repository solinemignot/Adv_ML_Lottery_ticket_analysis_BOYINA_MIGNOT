import pandas as pd
import time
from Neural_networks import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

############## Helper functions for comparison of initialization #######################################################################

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
    for pruning_round in range (rounds): 
        for df, df_avg in [(df_accuracies_method_1, df_avg_accuracies_method_1), (df_accuracies_method_2, df_avg_accuracies_method_2)]:
            df_round = df[df['Round'] == ("Initial model" if pruning_round == 0 else f"Round {pruning_round}")]
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




############## Helper functions for comparison of pruning #######################################################################

def comparing_pruning_methods(amount_of_repeats, rounds, method_1, method_2):
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
    for df, df_avg in [(df_accuracies_method_1, df_avg_accuracies_method_1), (df_accuracies_method_2, df_avg_accuracies_method_2)]:
        df_round = df[df['Round'].isin([f"Round {rounds-1}", "One_shot"])]
        round_avg_of_test_acc = df_round['Test Accuracy (with training)'].mean()
        min_test_acc = df_round['Test Accuracy (with training)'].min()
        max_test_acc = df_round['Test Accuracy (with training)'].max()
        pruning_perc = df_round['Pruning percentage'].mean()

        round_avg_info = {'Round' : 'End' , 
                        'Pruning Percentage' : pruning_perc,
                        'Avg Test Accuracy' : round_avg_of_test_acc,
                        'Min Test Accuracy' : min_test_acc,
                        'Max Test Accuracy' : max_test_acc}
        df_avg.append(round_avg_info)

    df_avg_accuracies_method_1 = pd.DataFrame(df_avg_accuracies_method_1)
    df_avg_accuracies_method_2 = pd.DataFrame(df_avg_accuracies_method_2)

    print((time.time()- beginning)/60)
    
    return df_avg_accuracies_method_1, df_avg_accuracies_method_2



def comparing_pruning_methods_plotting(df_avg_accuracies_method_1, df_avg_accuracies_method_2, rounds, method_1_name, method_2_name):
    df_final_method_1 = df_avg_accuracies_method_1[df_avg_accuracies_method_1['Round'] == "One_shot"]
    df_final_method_2 = df_avg_accuracies_method_2[df_avg_accuracies_method_2['Round'] == f"Round {rounds-1}"]

    # Add method names for plotting
    df_final_method_1['Method'] = method_1_name
    df_final_method_2['Method'] = method_2_name

    # Combine data for box plots
    df_final_combined = pd.concat([
        df_final_method_1[['Test Accuracy (with training)', 'Method']],
        df_final_method_2[['Test Accuracy (with training)', 'Method']]
    ], ignore_index=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Method', y='Test Accuracy (with training)', data=df_final_combined, palette="Set2")

    plt.title(f"Final Test Accuracies: {method_1_name} vs {method_2_name}")
    plt.xlabel("Pruning Method")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/comparing_pruning_methods_{method_1_name.replace(' ','')}_vs_{method_2_name.replace(' ','')}.png")
    plt.show()





