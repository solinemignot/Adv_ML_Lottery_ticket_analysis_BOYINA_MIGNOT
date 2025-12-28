import pandas as pd
import time
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Neural_networks import *
import re

############## Helper functions for comparison of initialization #################################

def clean(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def comparing_methods_initialization_after_pruning(amount_of_repeats, rounds, method_1, method_2):
    beginning_comparison = time.time()
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
            round_avg_info = {'Round' : round_name, 
                             'Pruning Percentage' : df_round['Pruning percentage'].mean()}
            for col in ['Test Accuracy (with training)', "Time (min)", "Final Training Loss"]:
                if not df_round.empty:
                    round_avg_info[f'Avg {col}'] = df_round[col].mean()
                    round_avg_info[f'Min {col}'] = df_round[col].min()
                    round_avg_info[f'Max {col}'] = df_round[col].max()
            df_avg.append(round_avg_info)

    df_avg_accuracies_method_1 = pd.DataFrame(df_avg_accuracies_method_1)
    df_avg_accuracies_method_2 = pd.DataFrame(df_avg_accuracies_method_2)

    print(f"Total time: {(time.time()- beginning_comparison)/60:.2f} minutes")
    
    return df_avg_accuracies_method_1, df_avg_accuracies_method_2


def comparing_methods_plotting(df1, df2, method_1_name, method_2_name, dataset_name, comp_col='Test Accuracy (with training)'):
    plt.figure(figsize=(10, 6))

    x1, y1 = df1['Pruning Percentage'], df1[f'Avg {comp_col}']
    x2, y2 = df2['Pruning Percentage'], df2[f'Avg {comp_col}']
    yerr1 = [df1[f'Avg {comp_col}'] - df1[f'Min {comp_col}'], df1[f'Max {comp_col}'] - df1[f'Avg {comp_col}']]
    yerr2 = [df2[f'Avg {comp_col}'] - df2[f'Min {comp_col}'], df2[f'Max {comp_col}'] - df2[f'Avg {comp_col}']]
    xerr1 = xerr2 = None
    xlabel = "Pruning Percentage (Sparsity)"

    if "time" in comp_col.lower():
        ylabel = 'Length of execution (minutes)'
    elif "loss" in comp_col.lower():
        ylabel = 'Final Training Loss'
    else:
        ylabel = 'Test Accuracy'

    plt.errorbar(x2, y2, xerr=xerr2, yerr=yerr2,
                 label=method_2_name, capsize=5, marker='x') #in this order to have LTH in orange to pop

    plt.errorbar(x1, y1, xerr=xerr1, yerr=yerr1,
                 label=method_1_name, capsize=5, marker='o')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Comparison: {method_1_name} vs {method_2_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots95/{clean(dataset_name)}_comparing_{clean(ylabel)}_{clean(method_1_name)}_vs_{clean(method_2_name)}.png")
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