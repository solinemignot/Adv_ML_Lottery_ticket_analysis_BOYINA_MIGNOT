import pandas as pd
import time
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Neural_networks import *

############## Helper functions for comparison of initialization #################################

def comparing_methods_initialization_after_pruning(amount_of_repeats, rounds, method_1, method_2):
    beginning = time.time()
    df_accuracies_method_1 = pd.DataFrame()
    df_accuracies_method_2 = pd.DataFrame()
    
    for i in tqdm(range(amount_of_repeats), desc="Repeats"):
        #method1
        df_acc_method_1, model_temp = method_1()
        del model_temp
        if torch.cuda.is_available(): torch.cuda.empty_cache() 
        
        for j in range(len(df_acc_method_1)):
            df_acc_method_1[j]['Iteration'] = i + 1
        df_accuracies_method_1 = pd.concat([df_accuracies_method_1, pd.DataFrame(df_acc_method_1)])

        #method2
        df_acc_method_2, model_temp = method_2()
        del model_temp
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        for j in range(len(df_acc_method_2)):
            df_acc_method_2[j]['Iteration'] = i + 1
        df_accuracies_method_2 = pd.concat([df_accuracies_method_2, pd.DataFrame(df_acc_method_2)])

    df_avg_accuracies_method_1 = []
    df_avg_accuracies_method_2 = []
    
    target_rounds = ["Initial model"] + [f"Round {r+1}" for r in range(rounds)] + ["One_shot"]
    
    for round_name in target_rounds:
        for df, df_avg in [(df_accuracies_method_1, df_avg_accuracies_method_1), (df_accuracies_method_2, df_avg_accuracies_method_2)]:
            
            df_round = df[df['Round'] == round_name]
            
            if not df_round.empty:
                round_avg_info = {
                    'Round' : round_name, 
                    'Pruning Percentage' : df_round['Pruning percentage'].mean(),
                    'Avg Test Accuracy' : df_round['Test Accuracy (with training)'].mean(),
                    'Min Test Accuracy' : df_round['Test Accuracy (with training)'].min(),
                    'Max Test Accuracy' : df_round['Test Accuracy (with training)'].max()
                }
                df_avg.append(round_avg_info)

    df_avg_accuracies_method_1 = pd.DataFrame(df_avg_accuracies_method_1)
    df_avg_accuracies_method_2 = pd.DataFrame(df_avg_accuracies_method_2)

    print(f"Total time: {(time.time()- beginning)/60:.2f} minutes")
    
    return df_avg_accuracies_method_1, df_avg_accuracies_method_2


def comparing_methods_plotting(df_avg_accuracies_method_1, df_avg_accuracies_method_2, method_1_name, method_2_name, dataset_name):
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(df_avg_accuracies_method_1['Pruning Percentage'],
                df_avg_accuracies_method_1['Avg Test Accuracy'],
                yerr=[df_avg_accuracies_method_1['Avg Test Accuracy'] - df_avg_accuracies_method_1['Min Test Accuracy'],
                        df_avg_accuracies_method_1['Max Test Accuracy'] - df_avg_accuracies_method_1['Avg Test Accuracy']],
                label = method_1_name, capsize=5, marker='o')
                
    plt.errorbar(df_avg_accuracies_method_2['Pruning Percentage'],
                df_avg_accuracies_method_2['Avg Test Accuracy'],
                yerr=[df_avg_accuracies_method_2['Avg Test Accuracy'] - df_avg_accuracies_method_2['Min Test Accuracy'],
                        df_avg_accuracies_method_2['Max Test Accuracy'] - df_avg_accuracies_method_2['Avg Test Accuracy']],
                label = method_2_name, capsize=5, marker='x')
    
    plt.xlabel("Pruning Percentage (Sparsity)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Comparison: {method_1_name} vs {method_2_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_comparing_{method_1_name}_vs_{method_2_name}.png")
    plt.show()


############## Helper functions for comparison of pruning (Iterative vs One Shot) #################################

def comparing_pruning_methods(amount_of_repeats, rounds, method_1, method_2):
    beginning = time.time()
    df_accuracies_method_1 = pd.DataFrame()
    df_accuracies_method_2 = pd.DataFrame()
    
    for i in tqdm(range(amount_of_repeats), desc="Repeats"):
        # Method 1
        df_acc_method_1, model_temp = method_1()
        del model_temp
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        for j in range(len(df_acc_method_1)):
            df_acc_method_1[j]['Iteration'] = i + 1
        df_accuracies_method_1 = pd.concat([df_accuracies_method_1, pd.DataFrame(df_acc_method_1)])

        # Method 2
        df_acc_method_2, model_temp = method_2()
        del model_temp
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        for j in range(len(df_acc_method_2)):
            df_acc_method_2[j]['Iteration'] = i + 1
        df_accuracies_method_2 = pd.concat([df_accuracies_method_2, pd.DataFrame(df_acc_method_2)])

    print(f"Total time: {(time.time()- beginning)/60:.2f} minutes")
    
    return df_accuracies_method_1, df_accuracies_method_2


def comparing_pruning_methods_plotting(df_accuracies_method_1, df_accuracies_method_2, rounds, method_1_name, method_2_name):
    df_final_method_1 = df_accuracies_method_1[df_accuracies_method_1['Round'] == f"Round {rounds}"]
    df_final_method_2 = df_accuracies_method_2[df_accuracies_method_2['Round'] == "One_shot"]
    
    if df_final_method_1.empty:
         df_final_method_1 = df_accuracies_method_1[df_accuracies_method_1['Round'] == "One_shot"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.boxplot(data=df_final_method_1, y='Test Accuracy (with training)', ax=axes[0])
    axes[0].set_ylabel('Test Accuracy') 
    axes[0].set_xlabel(method_1_name)  

    sns.boxplot(data=df_final_method_2, y='Test Accuracy (with training)', ax=axes[1])
    axes[1].set_ylabel('Test Accuracy')  
    axes[1].set_xlabel(method_2_name)      

    y_vals = pd.concat([df_final_method_1['Test Accuracy (with training)'], df_final_method_2['Test Accuracy (with training)']])
    if not y_vals.empty:
        y_min, y_max = y_vals.min(), y_vals.max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        axes[0].set_ylim(y_min - margin, y_max + margin)
        axes[1].set_ylim(y_min - margin, y_max + margin)

    fig.suptitle(f"Final Test Accuracies: {method_1_name} vs {method_2_name}")
    plt.tight_layout()
    plt.show()