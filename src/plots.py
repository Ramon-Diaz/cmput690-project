# %%
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
fig_size_wide = 10
fig_size_height = 8

import pathlib
'''
SNS - Sensitivity
PRE - Precision
F1 
SPC - Specificity
GMN - Geometric Mean Score

Undersampling
RUS Random Under Sampler
EdNN Edited Nearest Neighbours
ReEdNN Repeated Edited Nearest Neighbours
ConNN Condensed Nearest Neighbour
NCR Neighbourhood Cleaning Rule
OSS One sided Selection
IHT InstanceHardnessThreshold
CluCen Cluster Centroids
TomL Tomek Links

'''
# %%
def import_data(url):
    df = pd.read_csv(url)
    try:
        df['Oversampling Quantity'] = df['Oversampling Quantity'].str.rstrip('%').astype('float') / 100.0
    except KeyError:
        pass

    return df

def line_plot(df):

    temp = df.groupby(['Resampling','Oversampling Quantity']).mean()

    fig, ax = plt.subplots(figsize=(fig_size_wide, fig_size_height))
    ax.plot(temp['F1'].unstack(level=0), label=temp['F1'].unstack(level=0).columns.values)
    ax.set_title('F1 Score of Oversampling Techniques')
    ax.legend(loc='best')
    ax.set_ylabel('F1 score')
    ax.set_xlabel('Ratio')
    fig.tight_layout()

    fig.savefig(pathlib.Path.cwd().joinpath('..','plots','line_plot_f1_oversampling.pdf'), bbox_inches = 'tight')


def box_plot_by_oversampling(dfOver):

    df = pd.pivot_table(dfOver[['Resampling','Oversampling Quantity','F1']], values='F1',index='Oversampling Quantity', columns='Resampling')

    fig, ax = plt.subplots(figsize=(fig_size_wide, fig_size_height))
    bp1 = ax.boxplot(df, patch_artist=True)
    ax.set_title('F1 Score of Oversampling Techniques')
    colors = ['tab:blue', 'tab:orange', 'tab:green','tab:red','tab:purple','tab:brown','tab:pink']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp1['medians']:
        median.set(linewidth=3.0)
    ax.set_xticklabels(df.columns.values, rotation=75)
    ax.set_ylabel('F1 score')
    ax.set_xlabel('Oversampling Techniques')
    plt.grid(axis = 'y')
    fig.tight_layout()

    fig.savefig(pathlib.Path.cwd().joinpath('..','plots','box_plot_f1_oversampling.pdf'), bbox_inches = 'tight')

def box_plot_by_oversampling_algorithm(dfOver):

    df = pd.pivot_table(dfOver[['Model','Resampling','F1']], values='F1',index='Resampling', columns='Model').drop('AdaBoost',axis=1)

    fig, ax = plt.subplots(figsize=(fig_size_wide, fig_size_height))
    bp1 = ax.boxplot(df, patch_artist=True)
    ax.set_title('F1 Score of Classifier Techniques with Oversampling')
    colors = ['tab:blue', 'tab:orange', 'tab:green','tab:pink','tab:purple','tab:brown']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp1['medians']:
        median.set(color='red', linewidth=3.0)
    ax.set_xticklabels(df.columns.values, rotation=60)
    ax.set_ylabel('F1 score')
    ax.set_xlabel('Classifiers')
    plt.grid(axis = 'y')
    fig.tight_layout()

    fig.savefig(pathlib.Path.cwd().joinpath('..','plots','box_plot_f1_oversampling_classifiers.pdf'), bbox_inches = 'tight')



# %%
dfOver = import_data(pathlib.Path.cwd().joinpath('..','results','resultsOversampling_all.csv'))
dfUnder = import_data(pathlib.Path.cwd().joinpath('..','results','resultsUndersampling_all.csv'))
line_plot(dfOver)
box_plot_by_oversampling(dfOver)
box_plot_by_oversampling_algorithm(dfOver)
# %%
def box_plot_undersampling_algorithm(dfUnder):

    df = pd.pivot_table(dfUnder[['Model','Resampling','F1']], values='F1',index='Resampling', columns='Model')


    fig, ax = plt.subplots(figsize=(fig_size_wide, fig_size_height))
    bp1 = ax.boxplot(df, patch_artist=True)
    ax.set_title('F1 Score of Classifier Techniques with Oversampling')
    colors = ['tab:blue', 'tab:orange', 'tab:green','tab:pink','tab:purple','tab:brown']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp1['medians']:
        median.set(color='red', linewidth=3.0)
    ax.set_xticklabels(df.columns.values, rotation=60)
    ax.set_ylabel('F1 score')
    ax.set_xlabel('Classifiers')
    fig.tight_layout()
    plt.grid(axis = 'y')

    fig.savefig(pathlib.Path.cwd().joinpath('..','plots','box_plot_f1_undersampling_classifiers.pdf'), bbox_inches = 'tight')

def box_plot_undersampling_technique(dfUnder):
    df = pd.pivot_table(dfUnder.loc[dfUnder.Model != 'AdaBoost',['Model','Resampling','F1']], values='F1',index='Model', columns='Resampling')

    fig, ax = plt.subplots(figsize=(fig_size_wide, fig_size_height))
    bp1 = ax.boxplot(df, patch_artist=True)
    ax.set_title('F1 Score of Undersampling Techniques')
    colors = ['tab:blue', 'tab:orange', 'tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:grey','tab:cyan']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp1['medians']:
        median.set(linewidth=3.0)
    ax.set_xticklabels(df.columns.values, rotation=90)
    ax.set_ylabel('F1 score')
    ax.set_xlabel('Undersampling Techniques')
    ax.set_ylim([0.6, 0.95])
    fig.tight_layout()
    plt.grid(axis = 'y')

    fig.savefig(pathlib.Path.cwd().joinpath('..','plots','box_plot_f1_undersampling.pdf'), bbox_inches = 'tight')


# %%
box_plot_undersampling_algorithm(dfUnder)
box_plot_undersampling_technique(dfUnder)
# %%
