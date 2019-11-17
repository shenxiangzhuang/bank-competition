import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import PercentFormatter

# set matplot theme
mpl.style.use('fivethirtyeight')

# import data
trainX = pd.read_csv('../data/train.csv')
trainy = pd.read_csv('../data/train_target.csv')
testX = pd.read_csv('../data/test.csv')

########################
# 1.Data Preprocessing #
########################

# 1.1 Data Format
print('TrainX infomation:')
print(trainX.info())
print('TrainX columns: ', trainX.columns)
print('Trainy infomation:')
print(trainy.info())

# 1.2 Imbalance data[|N|/|P| = 136.673]
print(f'|N| / |P| = {sum(trainy.target == 0) / sum(trainy.target == 1)}')

# 1.3 missing data

# copy data and update it later
X = trainX.copy(deep=True)
y = trainy.copy(deep=True)

# find columns with missing data[only one: bankCard]
print(f'Missing columns: {trainX.columns[trainX.isna().sum(axis=0) != 0]}')
# preview the column
print(f'bankCard: \n{trainX.bankCard.head()}')

# transform: missing->bad(1); exist->good(0)
X.bankCard.replace(-999, 1, inplace=True)
X.bankCard.fillna(1, inplace=True)
X.loc[X.bankCard > 1, 'bankCard'] = 0
# do same on the test set
testX.bankCard.replace(-999, 1, inplace=True)
testX.bankCard.fillna(1, inplace=True)
testX.loc[testX.bankCard > 1, 'bankCard'] = 0

# 1.4 duplicated data[None]
print(X.duplicated().sum())


"""
Util now, we have handled the problems about missing data(just implicit,
-999 still there) and duplicated data both in training set and test set.

TODO: Plot to see the intrinsic information contain in the data.
"""

########################
# 2.Data Visualization #
########################

# columns[ignore x_1~x_78]
print([col for col in X.columns if 'x_' not in col])
# X,y -> df
df = pd.concat([X, y.target], axis=1)

# drop id and certId
df.drop(['id'], axis=1, inplace=True)
df.drop(['certId'], axis=1, inplace=True)
testX.drop(['certId'], axis=1, inplace=True)


# category plot
def catPlot(feature, save=False, bins=None):
    N = len(df[feature][df.target == 0])
    P = len(df[feature][df.target == 1])
    fig, axs = plt.subplots(1, 2,
                            tight_layout=True, figsize=(8, 4))

    axs[0].hist(df.loc[df.target == 0, feature],
                weights=np.ones(N) / N, bins=bins)
    axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[0].set_xlabel('Negative')
    axs[1].hist(df.loc[df.target == 1, feature],
                weights=np.ones(P) / P, bins=bins)
    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[1].set_xlabel('Positive')
    plt.suptitle(feature)
    if save:
        plt.savefig(f'../pics/{feature}.png', bbox_inches='tight', dpi=300)
    plt.show()


# loanProduct[plot]
catPlot('loanProduct', save=True)
# loanProduct[recode]
products = pd.get_dummies(pd.concat([df.loanProduct, testX.loanProduct],
                                    axis=0), prefix='product')
df = df.join(products.iloc[0:len(df), :])
df.drop(['loanProduct'], axis=1, inplace=True)
# test data
testX = testX.join(products.iloc[len(df):, :])
testX.drop(['loanProduct'], axis=1, inplace=True)

# gender
catPlot('gender', save=True)


# numeric plot
def numPlot(feature, save=False, filename=None):
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
    axs[0].boxplot(df.loc[df.target == 0, feature])
    axs[0].set_xlabel('Negative')
    axs[1].boxplot(df.loc[df.target == 1, feature])
    axs[1].set_xlabel('Positive')
    plt.suptitle(feature)
    if save:
        plt.savefig(f'../pics/{filename}.png', bbox_inches='tight', dpi=300)
    plt.show()


# age[standardization]
numPlot('age', save=True, filename='age[before]')
# remove outlier
df.drop(df.index[df.age > 100], inplace=True)
numPlot('age', save=True, filename='age[after]')

# standardization[WARN: data leakage]
age_mean = df.age.mean()
age_std = df.age.std()
# training data
df.age = (df.age - age_mean) / age_std
# test data
testX.age = (testX.age - age_mean) / age_std

# dist[drop]
df.drop(['dist'], axis=1, inplace=True)
testX.drop(['dist'], axis=1, inplace=True)

# edu[standardization]
df.drop(df.index[df.edu == -999], inplace=True)
edu_mean = df.edu.mean()
edu_std = df.edu.std()
df.edu = (df.edu - edu_mean) / edu_std
testX.edu = (testX.edu - edu_mean) / edu_std

# job[recode]
job_names = ['job_' + str(i) for i in range(1, 16)]
df[job_names] = pd.get_dummies(pd.concat([df.job, testX.job],
                                         axis=0)).iloc[0:len(df), :]
testX[job_names] = pd.get_dummies(pd.concat([df.job, testX.job],
                                            axis=0)).iloc[len(df):, :]

df.drop(['job'], axis=1, inplace=True)
testX.drop(['job'], axis=1, inplace=True)

# lmt[standardization]
lmt_mean = df.lmt.mean()
lmt_std = df.lmt.std()
df.lmt = (df.lmt - lmt_mean) / lmt_std
testX.lmt = (testX.lmt - lmt_mean) / lmt_std

# basicLevel[standardization]
df.basicLevel.replace(-999, -3, inplace=True)
testX.basicLevel.replace(-999, -3, inplace=True)
basicLevel_mean = df.basicLevel.mean()
basicLevel_std = df.basicLevel.std()
df.basicLevel = (df.basicLevel - basicLevel_mean) / basicLevel_std
testX.basicLevel = (testX.basicLevel - basicLevel_mean) / basicLevel_std

# certValidBegin, certValidStop[drop]
df.drop(['certValidBegin', 'certValidStop'], axis=1, inplace=True)
testX.drop(['certValidBegin', 'certValidStop'], axis=1, inplace=True)

# ethnic[recode]
# 汉族为0，其他为1
df.loc[df.ethnic == 0, 'ethnic'] = 0
df.loc[df.ethnic != 0, 'ethnic'] = 1
testX.loc[testX.ethnic == 0, 'ethnic'] = 0
testX.loc[testX.ethnic != 0, 'ethnic'] = 1

# residentAddr[drop]
df.drop(['residentAddr'], axis=1, inplace=True)
testX.drop(['residentAddr'], axis=1, inplace=True)

# highestEdu[standardization]
df.highestEdu.replace(-999, -90, inplace=True)
testX.highestEdu.replace(-999, -90, inplace=True)
highestEdu_mean = df.highestEdu.mean()
highestEdu_std = df.highestEdu.std()
# training data
df.highestEdu = (df.highestEdu - highestEdu_mean) / highestEdu_std
# test data
testX.highestEdu = (testX.highestEdu - highestEdu_mean) / highestEdu_std

# linkRela[recode]
relations = ['relation_' + str(i) for i in range(9)]
df[relations] = pd.get_dummies(pd.concat([df.linkRela, testX.linkRela],
                                         axis=0)).iloc[0:len(df), :]
testX[relations] = pd.get_dummies(pd.concat([df.linkRela, testX.linkRela],
                                            axis=0)).iloc[len(df):, :]
df.drop(['linkRela'], axis=1, inplace=True)
testX.drop(['linkRela'], axis=1, inplace=True)

# setupHour[recode]
df.loc[(df.setupHour >= 8) & (df.setupHour <= 17), 'setupHour'] = 0
df.loc[df.setupHour < 8, 'setupHour'] = 1
df.loc[df.setupHour > 17, 'setupHour'] = 1

testX.loc[(testX.setupHour >= 8) & (testX.setupHour <= 17), 'setupHour'] = 0
testX.loc[testX.setupHour < 8, 'setupHour'] = 1
testX.loc[testX.setupHour > 17, 'setupHour'] = 1

df.drop(['setupHour'], axis=1, inplace=True)
testX.drop(['setupHour'], axis=1, inplace=True)

# weekday[recode]
df.loc[df.weekday <= 5, 'weekday'] = 0
df.loc[df.weekday > 5, 'weekday'] = 1
testX.loc[testX.weekday <= 5, 'weekday'] = 0
testX.loc[testX.weekday > 5, 'weekday'] = 1
df.drop(['weekday'], axis=1, inplace=True)
testX.drop(['weekday'], axis=1, inplace=True)

# ncloseCreditCard...[fix]
spe_cols = ['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan',
            'unpayNormalLoan', '5yearBadloan']
df[spe_cols].head()

# isNew[drop]
df.drop(['isNew'], axis=1, inplace=True)
testX.drop(['isNew'], axis=1, inplace=True)

#################
# 3.Data Saving #
#################
# split
y = df.target
X = df.drop(['target'], axis=1)
# save
X.to_csv('../data/features/trainX.csv', index=None)
y.to_csv('../data/features/trainY.csv', index=None)
testX.to_csv('../data/features/testX.csv', index=None)
# (132029, 104) -> (132001, 119)
print(f'{X.shape} -> {trainX.shape}')
