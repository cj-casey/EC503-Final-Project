import argparse
# from NLP_BoW import BoW
from tfidf import tfidf,tfidf_BoW_input
import numpy as np
import pandas as pd
from FeatureSelection import chi_squared_selection, mutual_info_selection
from SVM_decision import svm
from RFC_decision import rfc
from NB_decision import nb

def pipeline(nlp, fs, model, dataset='20news', min_df=3, max_df=0.95, c=1, gamma=1, k=100):
    # settings
    # nlp = 'bow_tfidf','bow','tfidf'
    # fs = 'chi_square','mutual_info'
    # model = 'svm','rfc', 'nb'
    # dataset = '20news', addmore.....
    # hyperparameters
    # min_df = args.min_df # min df for tfidf
    # max_df = args.max_df # max df for tfidf
    # c = args.c # for SVC
    # gamma = args.gamma # for
    # k = args.topk
    # nlp = args.nlp

    # for cv_folds 0-4
    train_ccr = []
    train_f1_score = []
    train_conf_mat = []
    test_ccr = []
    test_f1_score = []
    test_conf_mat = []

    # Load the data
    if(nlp == 'bow'):
        print("Running BoW")
        # Load training data
        train_data = pd.read_csv("BoW_clickbait_train.csv")
        test_data = pd.read_csv("BoW_clickbait_test.csv")

        colTrain, colTest = len(train_data.columns), len(test_data.columns)
        minCol = min(colTrain, colTest)
        train_data = train_data.iloc[:, :minCol]
        test_data = test_data.iloc[:, :minCol]

        train_label = train_data['label']
        test_label = test_data['label']

        train_data = train_data.fillna(0)
        test_data = test_data.fillna(0)
        train_label = train_label.fillna(0)
        test_label = test_label.fillna(0)

        print("Train Data Shape:", train_data.shape)
        print("Train Label Shape:", train_label.shape)
        print("Test Data Shape:", test_data.shape)
        print("Test Label Shape:", test_label.shape)

    for cv_fold in range(0,4):
        if(nlp == 'bow'):
            if(cv_fold != 3):
                print(f"Skipping iteration {cv_fold} for BoW")
                continue
        elif(nlp == 'tfidf'):
            train_data, train_label, test_data, test_label = tfidf(cv_fold, min_df=min_df, max_df=max_df, save_csvs=False)
        else:
            print("Error: No NLP Method Entered")
            exit(-1)

        #call chi-square or mutual information based on settings
        if(fs =='chi_square'):
            print("Running Chi-Square")
            fs_train_data,fs_test_data = chi_squared_selection(nlp, cv_fold, train_data, train_label, test_data, test_label, k, False)
        elif(fs =='mutual_info'):
            print("Running Mutual Information")
            fs_train_data, fs_test_data = mutual_info_selection(nlp, cv_fold, train_data, train_label, test_data, test_label, k, False)
        elif(fs == 'no_feature_selection'):
            print("Running without Feature Selection")
            fs_train_data = train_data
            fs_test_data = test_data
        else:
            print("Error: No Feature Selection Method Entered")
            exit(-1)
        #call svm,rfc,xgb based on settings
        if(model == 'svm'):
            print("Running SVM")
            train_ccr_cv,train_f1_score_cv,train_conf_mat_cv,test_ccr_cv,test_f1_score_cv,test_conf_mat_cv = svm(fs_train_data,train_label,fs_test_data,test_label,gamma, c)
        elif(model == 'rfc'):
            print("Running RFC")
            train_ccr_cv,train_f1_score_cv,train_conf_mat_cv,test_ccr_cv,test_f1_score_cv,test_conf_mat_cv = rfc(fs_train_data, train_label, fs_test_data, test_label)
        elif(model == 'nb'):
            print("Running NB")
            train_ccr_cv,train_f1_score_cv,train_conf_mat_cv,test_ccr_cv,test_f1_score_cv,test_conf_mat_cv = nb(fs_train_data, train_label, fs_test_data, test_label)
        else:
            print("Error: No Model Entered")
            exit(-1)
        
        train_ccr.append(train_ccr_cv)
        train_f1_score.append(train_f1_score_cv)
        train_conf_mat.append(train_conf_mat_cv)
        test_ccr.append(test_ccr_cv)
        test_f1_score.append(test_f1_score_cv)
        test_conf_mat.append(test_conf_mat_cv)

    #compute average results
    print(f"--{nlp} {fs} {model}--")
    print(f"--TRAINING CCR--\n{np.mean(train_ccr)}")
    print(f"--TRAINING F1--\n{np.mean(train_f1_score)}")

    print(f"--TESTING CCR--\n{np.mean(test_ccr)}")
    print(f"--TESTING F1--\n{np.mean(test_f1_score)}")
    # Save results to a file
    with open('results.txt', 'a') as f:
        f.write(f"--{nlp} {fs} {model}--\n")
        f.write(f"--TRAINING CCR--\n{np.mean(train_ccr)}\n")
        f.write(f"--TRAINING F1--\n{np.mean(train_f1_score)}\n")
        f.write(f"--TESTING CCR--\n{np.mean(test_ccr)}\n")
        f.write(f"--TESTING F1--\n{np.mean(test_f1_score)}\n")
        f.write("\n")

if __name__ == "__main__":
    nlp_methods = ['bow']
    fs_methods = ['chi_square', 'mutual_info']
    # fs_methods = ['no_feature_selection']
    model_methods = ['svm', 'rfc', 'nb']

    for nlp in nlp_methods:
        for fs in fs_methods:
            for model in model_methods:
                print(f"Running {nlp} {fs} {model} pipeline")
                pipeline(nlp, fs, model, k=500)

    print("Done running all pipelines")
