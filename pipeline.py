import argparse
from NLP_BoW import BoW
from tfidf import tfidf,tfidf_BoW_input
import numpy as np
from FeatureSelection import chi_squared_selection, mutual_info_selection
from SVM_decision import svm
from RFC_decision import rfc
from XGB_decision import xgb

def main(args):
    # settings
    # nlp = 'bow_tfidf','bow','tfidf'
    # fs = 'chi_square','mutual_info'
    # model = 'svm','rfc','xgb'
    # dataset = '20news', addmore.....
    # hyperparameters
    min_df = args.min_df # min df for tfidf
    max_df = args.max_df # max df for tfidf
    c = args.c # for SVC
    gamma = args.gamma # for
    k = args.topk
    nlp = args.nlp

    # for cv_folds 0-4
    ccr = []
    f1_score = []
    conf_mat = []
    for cv_fold in range(0,4):
        #call BoW -> tfidf_BoW_input or BoW or tfidf() based on settings
        if(args.nlp == 'bow_tfidf'):
            train_data,train_label,test_data,test_label = BoW(cv_fold,args.dataset)
            train_data, train_label, test_data, test_label = tfidf_BoW_input(cv_fold, args.dataset)
        elif(args.nlp == 'bow'):
            train_data, train_label, test_data, test_label = BoW(cv_fold, args.dataset)
        elif(args.nlp == 'tfidf'):
            train_data, train_label, test_data, test_label = tfidf(cv_fold, args.dataset, min_df=min_df,
                                                               max_df=max_df, save_csvs=False)
        else:
            print("Error: No NLP Method Entered")
            exit(-1)

        #call chi-square or mutual information based on settings
        if(args.fs =='chi_square'):
            fs_train_data,fs_test_data = chi_squared_selection(nlp, train_data, train_label, test_data, test_label, k)
        elif(args.fs =='mutual_info'):
            fs_train_data, fs_test_data = mutual_info_selection(nlp, train_data, train_label, test_data, test_label, k)
        else:
            print("Error: No Feature Selection Method Entered")
            exit(-1)
        #call svm,rfc,xgb based on settings
        if(args.model == 'svm'):
            ccr[cv_fold],f1_score[cv_fold],conf_mat[cv_fold] = svm(fs_train_data,train_label,fs_test_data,test_label,gamma=gamma, c=c)
        elif(args.model == 'rfc'):
            ccr[cv_fold], f1_score[cv_fold], conf_mat[cv_fold] = rfc(fs_train_data, train_label, fs_test_data, test_label)
        elif(args.model =='xgb'):
            ccr[cv_fold], f1_score[cv_fold], conf_mat[cv_fold] = xgb(fs_train_data, train_label, fs_test_data, test_label)
        else:
            print("Error: No Model Entered")
            exit(-1)
        #compute average results
    print(f"--TESTING CCR--\n{np.mean(ccr)}")
    print(f"--TESTING F1--\n{np.mean(f1_score)}")
    

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A script that runs text classification pipeline with different settings")

    # pipeline settings
    parser.add_argument('-n', '--nlp', type=str, required=True, help='NLP Technique')
    parser.add_argument('-f', '--fs', type=str, required=True, help='Feature Selection Method')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Choice')
    parser.add_argument('-d', '--dataset', type=str, default='20news', help='Dataset Choice')
    # hyperparameters
    parser.add_argument('--min_df', type=int, required=False,default =3, help='Min_DF Hyperparameter')
    parser.add_argument('--max_df', type=float, required=False, default=0.95, help='max_df Hyperparameter')
    parser.add_argument('--c', type=float, required=False, default=1, help='c Hyperparameter')
    parser.add_argument('--gamma', type=float, required=False, default=1, help='gamma Hyperparameter')
    parser.add_argument('--topk', type=int, required=False, default=100, help='topk features Hyperparameter')
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)