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
            fs_train_data,fs_test_data = chi_squared_selection(nlp, cv_fold, train_data, train_label, test_data, test_label, k)
        elif(fs =='mutual_info'):
            fs_train_data, fs_test_data = mutual_info_selection(nlp, cv_fold, train_data, train_label, test_data, test_label, k)
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
    print(f"--TRAINING CCR--\n{np.mean(train_ccr)}")
    print(f"--TRAINING F1--\n{np.mean(train_f1_score)}")

    print(f"--TESTING CCR--\n{np.mean(test_ccr)}")
    print(f"--TESTING F1--\n{np.mean(test_f1_score)}")
    

if __name__ == "__main__":
    nlp_methods = ['tfidf']
    fs_methods = ['chi_square', 'mutual_info']
    model_methods = ['svm', 'rfc', 'nb']

    for nlp in nlp_methods:
        for fs in fs_methods:
            for model in model_methods:
                print(f"Running {nlp} {fs} {model} pipeline")
                pipeline(nlp, fs, model)

    print("Done running all pipelines")

# (myenv) PS C:\Users\alexa\OneDrive\Documents\Grad School\Fall 2024\EC503 Learning from Data\EC503-Final-Project> python pipeline.py
# Running tfidf chi_square svm pipeline
# Train Data Shape (15076, 5000)
# Train label shape (15076,)
# Test Data Shape (3770, 5000)
# Test label shape (3770,)
# Feature Selection using Chi-Squared for tfidf fold no. 0...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 96     windows  1154.019145  6.423964e-233
# 81        sale  1080.069337  4.185732e-217
# 31     clipper  1064.480339  8.980730e-214
# 17        bike   961.060940  1.081478e-191
# 90       space   959.049854  2.904043e-191
# 54      israel   954.444783  2.787732e-190
# 21         car   906.199647  5.376355e-180
# 47         god   889.227227  2.220063e-176
# 39  encryption   809.779352  1.793032e-159
# 52      hockey   751.398017  4.523062e-147
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 0 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 1...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1147.293624  1.764789e-231
# 81        sale  1133.294279  1.743297e-228
# 27     clipper  1056.743103  4.041594e-212
# 52      israel  1005.783829  3.091195e-201
# 15        bike   960.301728  1.570245e-191
# 91       space   940.518089  2.601652e-187
# 44         god   881.790017  8.519257e-175
# 18         car   841.705362  2.906266e-166
# 35  encryption   796.088890  1.457638e-156
# 53     israeli   743.015682  2.718527e-145
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 1 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 2...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1144.807441  6.005740e-231
# 29     clipper  1042.114873  5.391061e-209
# 81        sale  1018.734024  5.310802e-204
# 16        bike   966.892099  6.167101e-193
# 54      israel   948.769479  4.525418e-189
# 46         god   885.902899  1.133546e-175
# 19         car   877.147306  8.300630e-174
# 92       space   876.040186  1.428458e-173
# 37  encryption   820.200043  1.091056e-161
# 55     israeli   762.982813  1.570920e-149
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 2 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 3...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1198.112667  2.350704e-242
# 84        sale  1120.285175  1.056134e-225
# 29     clipper  1096.124594  1.548052e-220
# 16        bike  1016.409572  1.665300e-203
# 57      israel  1011.450242  1.907119e-202
# 91       space   980.631316  7.221465e-196
# 49         god   906.559932  4.505231e-180
# 19         car   851.751051  2.116575e-168
# 39  encryption   804.549104  2.319861e-158
# 58     israeli   751.987328  3.391185e-147
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 3 completed successfully!
# --TRAINING CCR--
# 0.5383288510506772
# --TRAINING F1--
# 0.5964617333437714
# --TESTING CCR--
# 0.5343901069242101
# --TESTING F1--
# 0.5912902042975908
# Running tfidf chi_square rfc pipeline
# Train Data Shape (15076, 5000)
# Train label shape (15076,)
# Test Data Shape (3770, 5000)
# Test label shape (3770,)
# Feature Selection using Chi-Squared for tfidf fold no. 0...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 96     windows  1154.019145  6.423964e-233
# 81        sale  1080.069337  4.185732e-217
# 31     clipper  1064.480339  8.980730e-214
# 17        bike   961.060940  1.081478e-191
# 90       space   959.049854  2.904043e-191
# 54      israel   954.444783  2.787732e-190
# 21         car   906.199647  5.376355e-180
# 47         god   889.227227  2.220063e-176
# 39  encryption   809.779352  1.793032e-159
# 52      hockey   751.398017  4.523062e-147
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 0 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 1...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1147.293624  1.764789e-231
# 81        sale  1133.294279  1.743297e-228
# 27     clipper  1056.743103  4.041594e-212
# 52      israel  1005.783829  3.091195e-201
# 15        bike   960.301728  1.570245e-191
# 91       space   940.518089  2.601652e-187
# 44         god   881.790017  8.519257e-175
# 18         car   841.705362  2.906266e-166
# 35  encryption   796.088890  1.457638e-156
# 53     israeli   743.015682  2.718527e-145
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 1 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 2...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1144.807441  6.005740e-231
# 29     clipper  1042.114873  5.391061e-209
# 81        sale  1018.734024  5.310802e-204
# 16        bike   966.892099  6.167101e-193
# 54      israel   948.769479  4.525418e-189
# 46         god   885.902899  1.133546e-175
# 19         car   877.147306  8.300630e-174
# 92       space   876.040186  1.428458e-173
# 37  encryption   820.200043  1.091056e-161
# 55     israeli   762.982813  1.570920e-149
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 2 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 3...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1198.112667  2.350704e-242
# 84        sale  1120.285175  1.056134e-225
# 29     clipper  1096.124594  1.548052e-220
# 16        bike  1016.409572  1.665300e-203
# 57      israel  1011.450242  1.907119e-202
# 91       space   980.631316  7.221465e-196
# 49         god   906.559932  4.505231e-180
# 19         car   851.751051  2.116575e-168
# 39  encryption   804.549104  2.319861e-158
# 58     israeli   751.987328  3.391185e-147
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 3 completed successfully!
# --TRAINING CCR--
# 0.7622994798541275
# --TRAINING F1--
# 0.8209734938821276
# --TESTING CCR--
# 0.5639049505494003
# --TESTING F1--
# 0.5944814337997575
# Running tfidf chi_square nb pipeline
# Train Data Shape (15076, 5000)
# Train label shape (15076,)
# Test Data Shape (3770, 5000)
# Test label shape (3770,)
# Feature Selection using Chi-Squared for tfidf fold no. 0...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 96     windows  1154.019145  6.423964e-233
# 81        sale  1080.069337  4.185732e-217
# 31     clipper  1064.480339  8.980730e-214
# 17        bike   961.060940  1.081478e-191
# 90       space   959.049854  2.904043e-191
# 54      israel   954.444783  2.787732e-190
# 21         car   906.199647  5.376355e-180
# 47         god   889.227227  2.220063e-176
# 39  encryption   809.779352  1.793032e-159
# 52      hockey   751.398017  4.523062e-147
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 0 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 1...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1147.293624  1.764789e-231
# 81        sale  1133.294279  1.743297e-228
# 27     clipper  1056.743103  4.041594e-212
# 52      israel  1005.783829  3.091195e-201
# 15        bike   960.301728  1.570245e-191
# 91       space   940.518089  2.601652e-187
# 44         god   881.790017  8.519257e-175
# 18         car   841.705362  2.906266e-166
# 35  encryption   796.088890  1.457638e-156
# 53     israeli   743.015682  2.718527e-145
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 1 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 2...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1144.807441  6.005740e-231
# 29     clipper  1042.114873  5.391061e-209
# 81        sale  1018.734024  5.310802e-204
# 16        bike   966.892099  6.167101e-193
# 54      israel   948.769479  4.525418e-189
# 46         god   885.902899  1.133546e-175
# 19         car   877.147306  8.300630e-174
# 92       space   876.040186  1.428458e-173
# 37  encryption   820.200043  1.091056e-161
# 55     israeli   762.982813  1.570920e-149
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 2 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Chi-Squared for tfidf fold no. 3...
# Labels reshaped to 1D successfully!
# Computing Chi-Squared scores...
# Chi-Squared scores computed successfully!
# Chi-Squared features selected successfully!
# Top 10 of 100 selected features with Chi-Squared:
#        Feature   Chi2_Score        P_Value
# 97     windows  1198.112667  2.350704e-242
# 84        sale  1120.285175  1.056134e-225
# 29     clipper  1096.124594  1.548052e-220
# 16        bike  1016.409572  1.665300e-203
# 57      israel  1011.450242  1.907119e-202
# 91       space   980.631316  7.221465e-196
# 49         god   906.559932  4.505231e-180
# 19         car   851.751051  2.116575e-168
# 39  encryption   804.549104  2.319861e-158
# 58     israeli   751.987328  3.391185e-147
# Transformed Chi-Squared training dataset saved successfully!
# Transformed Chi-Squared test dataset saved successfully!
# Feature Selection using Chi-Squared for tfidf fold 3 completed successfully!
# --TRAINING CCR--
# 0.5458237811620696
# --TRAINING F1--
# 0.5658059745107284
# --TESTING CCR--
# 0.5455330481176539
# --TESTING F1--
# 0.5648720570193997
# Running tfidf mutual_info svm pipeline
# Train Data Shape (15076, 5000)
# Train label shape (15076,)
# Test Data Shape (3770, 5000)
# Test label shape (3770,)
# Feature Selection using Mutual Information for tfidf fold no. 0...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 98     windows  0.108187
# 38         god  0.097424
# 87        team  0.082785
# 22     clipper  0.081242
# 13         car  0.080392
# 79        sale  0.079239
# 39  government  0.078486
# 18   christian  0.077941
# 99      writes  0.074024
# 84       space  0.068615
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 0 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 1...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 98     windows  0.106231
# 41         god  0.099488
# 79        sale  0.088430
# 26     clipper  0.088230
# 42  government  0.085995
# 99      writes  0.076386
# 78     rutgers  0.073440
# 19   christian  0.072597
# 89        team  0.070525
# 13         car  0.070029
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 1 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 2...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 96     windows  0.100953
# 37         god  0.095838
# 98      writes  0.088514
# 75        sale  0.085614
# 18     clipper  0.083587
# 85        team  0.076314
# 38  government  0.075943
# 10         car  0.075003
# 61      people  0.072965
# 35        game  0.070903
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 2 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 3...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 96     windows  0.122547
# 46         god  0.101789
# 80        sale  0.092394
# 98      writes  0.084552
# 24     clipper  0.083126
# 15         car  0.081879
# 84       space  0.078004
# 47  government  0.077748
# 71      people  0.074643
# 78     rutgers  0.073868
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 3 completed successfully!
# --TRAINING CCR--
# 0.5417117145253815
# --TRAINING F1--
# 0.5750685969809731
# --TESTING CCR--
# 0.5306088761240132
# --TESTING F1--
# 0.5645877215108461
# Running tfidf mutual_info rfc pipeline
# Train Data Shape (15076, 5000)
# Train label shape (15076,)
# Test Data Shape (3770, 5000)
# Test label shape (3770,)
# Feature Selection using Mutual Information for tfidf fold no. 0...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 43         god  0.118277
# 97     windows  0.105474
# 81        sale  0.100120
# 13         car  0.086621
# 19   christian  0.078374
# 23     clipper  0.076912
# 87       space  0.076570
# 98      writes  0.073870
# 44  government  0.072670
# 51        host  0.070100
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 0 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 1...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 40         god  0.115293
# 97     windows  0.112168
# 77        sale  0.081818
# 98      writes  0.079472
# 22     clipper  0.076972
# 88        team  0.075952
# 38        game  0.075301
# 12        bike  0.074754
# 41  government  0.074164
# 67      people  0.072488
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 1 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 2...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 43         god  0.108694
# 96     windows  0.103887
# 25     clipper  0.087954
# 97      writes  0.084315
# 16         car  0.077577
# 79        sale  0.076641
# 87        team  0.073827
# 41        game  0.073788
# 53       jesus  0.072206
# 44  government  0.071766
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 2 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 3...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 41         god  0.112546
# 97     windows  0.111008
# 78        sale  0.084378
# 82       space  0.082401
# 14         car  0.080950
# 42  government  0.079041
# 25     clipper  0.076669
# 67      people  0.076392
# 98      writes  0.075761
# 20   christian  0.073561
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 3 completed successfully!
# --TRAINING CCR--
# 0.9949922603017785
# --TRAINING F1--
# 0.9950933224848246
# --TESTING CCR--
# 0.5944153336622299
# --TESTING F1--
# 0.5943851922970863
# Running tfidf mutual_info nb pipeline
# Train Data Shape (15076, 5000)
# Train label shape (15076,)
# Test Data Shape (3770, 5000)
# Test label shape (3770,)
# Feature Selection using Mutual Information for tfidf fold no. 0...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 97     windows  0.117132
# 43         god  0.097194
# 44  government  0.086851
# 15         car  0.083847
# 24     clipper  0.082907
# 98      writes  0.081977
# 78        sale  0.081074
# 86        team  0.078922
# 40        game  0.074464
# 20   christian  0.071202
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 0 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 1...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 97     windows  0.113617
# 36         god  0.101812
# 78        sale  0.096483
# 98      writes  0.088775
# 34        game  0.082096
# 37  government  0.080262
# 63      people  0.075848
# 21     clipper  0.074530
# 15        chip  0.071956
# 85        team  0.071918
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 1 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 2...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 96     windows  0.118481
# 38         god  0.110584
# 80        sale  0.089581
# 21     clipper  0.087585
# 98      writes  0.081702
# 66      people  0.076982
# 36        game  0.075589
# 39  government  0.073717
# 89        team  0.070808
# 30  encryption  0.067239
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 2 completed successfully!
# Train Data Shape (15077, 5000)
# Train label shape (15077,)
# Test Data Shape (3769, 5000)
# Test label shape (3769,)
# Feature Selection using Mutual Information for tfidf fold no. 3...
# Labels reshaped to 1D successfully!
# Computing Mutual Information scores...
# Mutual Information scores computed successfully!
# Mutual Information features selected successfully!
# Top 10 of 100 selected features with Mutual Information:
#        Feature  MI_Score
# 96     windows  0.111746
# 41         god  0.105895
# 89        team  0.090926
# 97      writes  0.089854
# 81        sale  0.085510
# 42  government  0.079671
# 21     clipper  0.074891
# 52       jesus  0.071234
# 39        game  0.071219
# 79     rutgers  0.071201
# Transformed Mutual Information training dataset saved successfully!
# Transformed Mutual Information test dataset saved successfully!
# Feature Selection using Mutual Information for tfidf fold no. 3 completed successfully!
# --TRAINING CCR--
# 0.5456913454508301
# --TRAINING F1--
# 0.5381846780792586
# --TESTING CCR--
# 0.5381038107188829
# --TESTING F1--
# 0.5303086077915529
# Done running all pipelines
# (myenv) PS C:\Users\alexa\OneDrive\Documents\Grad School\Fall 2024\EC503 Learning from Data\EC503-Final-Project>