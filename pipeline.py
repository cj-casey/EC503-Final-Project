import argparse
# from NLP_BoW import BoW
from tfidf import tfidf,tfidf_BoW_input
import numpy as np
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
    for cv_fold in range(0,4):
        #call BoW -> tfidf_BoW_input or BoW or tfidf() based on settings
        if(nlp == 'bow_tfidf'):
            train_data,train_label,test_data,test_label = BoW(cv_fold,dataset)
            train_data, train_label, test_data, test_label = tfidf_BoW_input(cv_fold, min_df=min_df, max_df=max_df, save_csvs=False)
        elif(nlp == 'bow'):
            train_data, train_label, test_data, test_label = BoW(cv_fold)
        elif(nlp == 'tfidf'):
            train_data, train_label, test_data, test_label = tfidf(cv_fold, min_df=min_df, max_df=max_df, save_csvs=False)
        else:
            print("Error: No NLP Method Entered")
            exit(-1)

        #call chi-square or mutual information based on settings
        if(fs =='chi_square'):
            fs_train_data,fs_test_data = chi_squared_selection(nlp, cv_fold, train_data, train_label, test_data, test_label, k, False)
        elif(fs =='mutual_info'):
            fs_train_data, fs_test_data = mutual_info_selection(nlp, cv_fold, train_data, train_label, test_data, test_label, k, False)
        elif(fs == 'no_feature_selection'):
            fs_train_data = train_data
            fs_test_data = test_data
        else:
            print("Error: No Feature Selection Method Entered")
            exit(-1)
        #call svm,rfc,xgb based on settings
        if(model == 'svm'):
            train_ccr_cv,train_f1_score_cv,train_conf_mat_cv,test_ccr_cv,test_f1_score_cv,test_conf_mat_cv = svm(fs_train_data,train_label,fs_test_data,test_label,gamma, c)
        elif(model == 'rfc'):
            train_ccr_cv,train_f1_score_cv,train_conf_mat_cv,test_ccr_cv,test_f1_score_cv,test_conf_mat_cv = rfc(fs_train_data, train_label, fs_test_data, test_label)
        elif(model == 'nb'):
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
    nlp_methods = ['tfidf']
    # fs_methods = ['chi_square', 'mutual_info']
    fs_methods = ['no_feature_selection']
    model_methods = ['svm', 'rfc', 'nb']

    for nlp in nlp_methods:
        for fs in fs_methods:
            for model in model_methods:
                print(f"Running {nlp} {fs} {model} pipeline")
                pipeline(nlp, fs, model, k=1500)

    print("Done running all pipelines")
