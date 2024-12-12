import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast


def plot_scores():
    # comment and uncomment according to the way you want to plot
    #read data
    #results = pd.read_csv('results.csv')
    # results = pd.read_csv('results_clickbait.csv')
    #results = pd.read_csv('bow_results.csv')
    results = pd.read_csv('bow_results_clickbait.csv')
    results_d = results.sort_values(by='Testing F1')
    configs = list(results_d['Configuration'])
    testing_f1_scores = list(results_d['Testing F1'])
    norm = plt.Normalize(min(testing_f1_scores), max(testing_f1_scores))
    cmap = plt.colormaps['turbo']
    colors = cmap(norm(testing_f1_scores))
    fig, ax = plt.subplots()
    ax.bar(configs, testing_f1_scores, color=colors)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Pipeline Configurations')
    ax.set_ylabel('F1-Score')
    ax.set_title('Clickbait BoW - Pipeline Performance - Testing')
    plt.savefig('testing_f1_bar_plot.png')

    #make traiing f1 plot
    results_a = results.sort_values(by = 'Testing F1')
    configs = list(results_a['Configuration'])
    training_f1_scores = list(results_a['Training F1'])
    #make different colors
    colors = cmap(norm(testing_f1_scores))
    fig,ax = plt.subplots()
    ax.bar(configs,training_f1_scores,color=colors)
    ax.tick_params(axis='x',labelsize=8)
    ax.set_xlabel('Pipeline Configurations')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1)
    ax.set_title('Pipeline Configuration Performance - Training')
    plt.savefig('training_f1_bar_plot.png')

    results_c = results.sort_values(by='Testing CCR')
    configs = list(results_c['Configuration'])
    testing_ccr_scores = list(results_c['Testing CCR'])
    fig, ax = plt.subplots()
    norm = plt.Normalize(min(testing_ccr_scores), max(testing_ccr_scores))
    cmap = plt.colormaps['turbo']
    colors = cmap(norm(testing_ccr_scores))
    ax.bar(configs, testing_ccr_scores, color=colors)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Pipeline Configurations')
    ax.set_ylabel('CCR')
    ax.set_ylim(0, 1)
    ax.set_title('20newsgroup TFIDF - Pipeline Performance')
    plt.savefig('testing_ccr_bar_plot.png')

    results_b = results.sort_values(by='Testing CCR')
    configs = list(results_b['Configuration'])
    training_ccr_scores = list(results_b['Training CCR'])
    # make different colors
    norm = plt.Normalize(min(testing_ccr_scores), max(testing_ccr_scores))
    cmap = plt.colormaps['turbo']
    colors = cmap(norm(testing_ccr_scores))
    fig, ax = plt.subplots()
    ax.bar(configs, training_ccr_scores, color=colors)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Pipeline Configurations')
    ax.set_ylabel('CCR')
    ax.set_ylim(0, 1)
    ax.set_title('Clickbait - Pipeline Performance - Training')
    plt.savefig('training_ccr_bar_plot.png')

def plot_hyperparameters():
    results = pd.read_csv('feature_select_pipeline_results.csv')
    feature = results['k_features_selected'].values.tolist()
    print(feature)
    f1 = results['test_f1_score'].values.tolist()
    print(f1)
    fig, ax = plt.subplots()
    ax.plot(feature, f1)
    ax.tick_params(axis='x',color='red')
    ax.axhline(0.8816980476406753,color='red',linestyle='--')
    ax.legend(labels=['tfidf_chi_svm', 'tfidf_none_svm'])
    ax.set_ylim(0.6, 1)
    ax.set_xlabel('Top K Features')
    ax.set_ylabel('F1-Score')
    ax.set_title('20Newsgroups TFIDF - Top K-Features vs F1-Score')
    plt.savefig('k_hyperparameter_plot.png')


# def plot_hyperparameters():
# plot_hyperparameters()
plot_scores()