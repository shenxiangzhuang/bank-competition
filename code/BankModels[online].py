#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ## Prepare
get_ipython().system('pip install algo-timer')

# import lib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('fivethirtyeight')
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from sklearn.metrics import make_scorer
import time
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
import os
import gc
from algotimer import Timer, TimerPloter
import pandas as pd

testX = pd.read_csv("../input/testX.csv")
X = pd.read_csv("../input/trainX.csv")
y = pd.read_csv("../input/trainY.csv", header=None)

ids = testX.id
testX = testX.drop(['id'], axis=1)

# ROC
def plotROC(y_test, y_score, pltName):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[2], tpr[2], _ = roc_curve(y_test, y_score)
    roc_auc[2] = auc(fpr[2], tpr[2])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # plot it
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='AUC=%0.2f' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC of ' + pltName)
    plt.legend(loc="best")
    plt.savefig(f'{pltName}.png', bbox_inches='tight', dpi=300)
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# ## XGBoost
def objective(params):
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 5
    count = 1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    score_mean = 0
    for tr_idx, val_idx in skf.split(X_train, y_train.values.ravel()):
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=300, random_state=4, verbose=True, 
            tree_method='hist', 
            scale_pos_weight=136,
            n_jobs=-1,
            **params
        )

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr.values.ravel())
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

space = {
    # The maximum depth of a tree, same as GBM.
    # Used to control over-fitting as higher depth will allow model 
    # to learn relations very specific to a particular sample.
    # Should be tuned using CV.
    # Typical values: 3-10
    'max_depth': hp.quniform('max_depth', 2, 6, 1),
    
    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 
    # (meaning pulling weights to 0). It can be more useful when the objective
    # is logistic regression since you might need help with feature selection.
    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    
    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
    # approach can be more useful in tree-models where zeroing 
    # features might not make much sense.
    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.4),
    
    # eta: Analogous to learning rate in GBM
    # Makes the model more robust by shrinking the weights on each step
    # Typical final values to be used: 0.01-0.2
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    
    # colsample_bytree: Similar to max_features in GBM. Denotes the 
    # fraction of columns to be randomly samples for each tree.
    # Typical values: 0.5-1
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    
    # A node is split only when the resulting split gives a positive
    # reduction in the loss function. Gamma specifies the 
    # minimum loss reduction required to make a split.
    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    'gamma': hp.uniform('gamma', 0.01, .7),
    
    # more increases accuracy, but may lead to overfitting.
    # num_leaves: the number of leaf nodes to use. Having a large number 
    # of leaves will improve accuracy, but will also lead to overfitting.
    'num_leaves': hp.choice('num_leaves', list(range(10, 100, 10))),
    
    # specifies the minimum samples per leaf node.
    # the minimum number of samples (data) to group into a leaf. 
    # The parameter can greatly assist with overfitting: larger sample
    # sizes per leaf will reduce overfitting (but may lead to under-fitting).
    'min_child_samples': hp.choice('min_child_samples', list(range(10, 200, 20))),
    
    # subsample: represents a fraction of the rows (observations) to be 
    # considered when building each subtree. Tianqi Chen and Carlos Guestrin
    # in their paper A Scalable Tree Boosting System recommend 
    'subsample': hp.choice('subsample', [0.7, 0.8, 0.9, 1]),
    
    # randomly select a fraction of the features.
    # feature_fraction: controls the subsampling of features used
    # for training (as opposed to subsampling the actual training data in 
    # the case of bagging). Smaller fractions reduce overfitting.
    'feature_fraction': hp.uniform('feature_fraction', 0.7, 1),
    
    # randomly bag or subsample training data.
    'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 1)
    
    # bagging_fraction and bagging_freq: enables bagging (subsampling) 
    # of the training data. Both values need to be set for bagging to be used.
    # The frequency controls how often (iteration) bagging is used. Smaller
    # fractions and frequencies reduce overfitting.
}

# Set algoritm parameters
with Timer('XGBoost, Search') as t:
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=20)
    best_params = space_eval(space, best)


# Print best parameters
best_params['max_depth'] = int(best_params['max_depth'])
print("BEST PARAMS: ", best_params)

clf = xgb.XGBClassifier(
    n_estimators=300,
    **best_params,
    tree_method='hist',
    eval_metric="auc",
    n_jobs=-1,
    scale_pos_weight=136
)

clf.fit(X_train, y_train.values.ravel())

xgb_y_train_pred = clf.predict_proba(X_train)[:,1]
plotROC(y_train, xgb_y_train_pred, 'XGBoost-Train')

xgb_y_test_pred = clf.predict_proba(X_test)[:,1]
plotROC(y_test, xgb_y_test_pred, 'XGBoost-Test')

# train with all data
with Timer('XGBoost, Train') as t:
    clf.fit(X, y.values.ravel(), verbose=1)

# train auc
xgb_y_all_pred = clf.predict_proba(X)[:,1]
roc_auc_score(y, xgb_y_all_pred)

plotROC(y, xgb_y_all_pred, 'XGBoost-Train-AllData')

result = pd.DataFrame()
result['id'] = ids
with Timer('XGBoost, Prediction') as t:
    result['target'] = clf.predict_proba(testX)[:,1]
result.to_csv('xgb.csv', index=None)


# ## Easy Ensemble Classifier
def objectiveEasy(params):
    time1 = time.time()
    params = {
        'sampling_strategy': params['sampling_strategy'],
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 5
    count = 1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    score_mean = 0
    for tr_idx, val_idx in skf.split(X_train, y_train.values.ravel()):
        clf = EasyEnsembleClassifier(**params,
                                    random_state=0,
                                    n_estimators=300,
                                    n_jobs=-1,
                                    verbose=0)

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr.values.ravel())
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

spaceEasy = {
    'sampling_strategy': hp.choice('sampling_strategy', [0.7, 0.8, 0.9, 1.0, 'auto'])
}

# Set algoritm parameters
with Timer('EasyEnsamble, Search') as t:
    bestEasy = fmin(fn=objectiveEasy,
                space=spaceEasy,
                algo=tpe.suggest,
                max_evals=5)

    # Print best parameters
    bestEasy_params = space_eval(spaceEasy, bestEasy)

bestEasy_params

clf = EasyEnsembleClassifier(**bestEasy_params,
                            random_state=0,
                            n_estimators=300,
                            n_jobs=-1,
                            verbose=1)

clf.fit(X_train, y_train)

# training roc
easy_y_train_pred = clf.predict_proba(X_train)[:,1]
plotROC(y_train, easy_y_train_pred, 'EasyEnsamble-Train')
# test roc
easy_y_test_pred = clf.predict_proba(X_test)[:,1]
plotROC(y_test, easy_y_test_pred, 'EasyEnsamble-Test')

# fit all data
with Timer('EasyEnsamble, Train') as t:
    clf.fit(X, y.values.ravel())

easy_y_all_pred = clf.predict_proba(X)[:, 1]
plotROC(y, easy_y_all_pred, 'EasyEnsamble-Train-AllData')
roc_auc_score(y, easy_y_all_pred)

# pridict
result = pd.DataFrame()
result['id'] = ids
with Timer('EasyEnsamble, Prediction') as t:
    result['target'] = clf.predict_proba(testX)[:, 1]
result.to_csv('EasyEnsemble.csv', index=None)


# ## Balanced Bagging Classifier
def objectiveBalance(params):
    time1 = time.time()
    params = {
        'sampling_strategy': params['sampling_strategy'],
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 5
    count = 1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    score_mean = 0
    for tr_idx, val_idx in skf.split(X_train, y_train.values.ravel()):
        clf = BalancedBaggingClassifier(**params,
                                    random_state=0,
                                    n_estimators=300,
                                    n_jobs=-1,
                                    verbose=0)

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr.values.ravel())
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

spaceBalance = spaceEasy

with Timer('BalancedEnsamble, Search') as t:
    # Set algoritm parameters
    bestBalance = fmin(fn=objectiveBalance,
                space=spaceBalance,
                algo=tpe.suggest,
                max_evals=5)
  
    # Print best parameters
    bestBalance_params = space_eval(spaceBalance, bestBalance)

clf = BalancedBaggingClassifier(**bestBalance_params,
                                random_state=0,
                                n_estimators=300,
                                n_jobs=-1,
                                verbose=0)

clf.fit(X_train, y_train)

# training roc
balance_y_train_pred = clf.predict_proba(X_train)[:,1]
plotROC(y_train, balance_y_train_pred, 'BalancedEnsamble-Train')
# test roc
balance_y_test_pred = clf.predict_proba(X_test)[:,1]
plotROC(y_test, balance_y_test_pred, 'BalancedEnsamble-Test')

with Timer('BalancedEnsamble, Train') as t:
    # train with all data
    clf.fit(X, y.values.ravel())

balance_y_all_pred = clf.predict_proba(X)[:, 1]
plotROC(y, balance_y_all_pred, 'BalancedEnsamble-Train-AllData')
roc_auc_score(y, balance_y_all_pred)

result = pd.DataFrame()
result['id'] = ids
with Timer('BalancedEnsamble, Prediction') as t:
    result['target'] = clf.predict_proba(testX)[:, 1]
result.to_csv('BalancedBaggingClassifier.csv', index=None)

def subplotRoc(y_test, y_score, pltName):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[2], tpr[2], _ = roc_curve(y_test, y_score)
    roc_auc[2] = auc(fpr[2], tpr[2])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # plot it
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='AUC=%0.2f' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC of ' + pltName)
    plt.legend(loc="best")

# 3 x 3 ROC plot
fig = plt.subplots(3, 3, figsize=(16, 14))
plt.subplots_adjust(hspace=0.6)
plt.suptitle('ROC of All')
plt.subplot(331)
subplotRoc(y_train, xgb_y_train_pred, 'XGBoost-Train')
plt.subplot(332)
subplotRoc(y_test, xgb_y_test_pred, 'XGBoost-Test')
plt.subplot(333)
subplotRoc(y, xgb_y_all_pred, 'XGBoost-Train-All-Data')
plt.subplot(334)
subplotRoc(y_train, easy_y_train_pred, 'EasyEnsamble-Train')
plt.subplot(335)
subplotRoc(y_test, easy_y_test_pred, 'EasyEnsamble-Test')
plt.subplot(336)
subplotRoc(y, easy_y_all_pred, 'EasyEnsamble-Train-AllData')
plt.subplot(337)
subplotRoc(y_train, balance_y_train_pred, 'BalancedEnsamble-Train')
plt.subplot(338)
subplotRoc(y_test, balance_y_test_pred, 'BalancedEnsamble-Test')
plt.subplot(339)
subplotRoc(y, balance_y_all_pred, 'BalancedEnsamble-Train-AllData')

plt.savefig('ROC.png', bbox_inches='tight', dpi=300)
plt.show()
