#!/usr/bin/env python
# coding: utf-8

# ## 探索性数据分析import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from scipy import stats
from matplotlib.ticker import PercentFormatter
mpl.style.use('fivethirtyeight')
# from sklearn.decomposition import SparsePCA, PCA
# import data
trainX = pd.read_csv('../data/train.csv')
trainy = pd.read_csv('../data/train_target.csv')
testX = pd.read_csv('../data/test.csv')


# ### 数据预处理

# #### 数据格式辅助函数# tools function
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    return summary


# #### 数据格式trainX.info()
trainX.columns
testX.info()
trainy.info()
resumetable(trainX)


# #### 类别不平衡问题sum(trainy.target == 0) / sum(trainy.target == 1)
total = len(trainX)
plt.figure(figsize=(10,6), tight_layout=True)
g = sns.countplot(x='target', data=pd.concat([trainX, trainy.target], axis=1))
g.set_title("Fraud Transactions Distribution \n# 0: No Fraud | 1: Fraud #", fontsize=22)
g.set_xlabel("Is fraud?", fontsize=18)
g.set_ylabel('Count', fontsize=18)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15)
plt.savefig(f'../pics/imbalance.png', bbox_inches='tight', dpi=300)
plt.show()


# #### 数据重复值与缺失值# copy，update
X = trainX
y = trainy
# 含有缺失值的列
trainX.columns[trainX.isna().sum(axis=0) != 0]
trainX.bankCard.head()
trainX.bankCard.isna().sum()
sum(trainX.bankCard == -999)
# "bad" samples
X.bankCard.replace(-999, 1, inplace=True)
X.bankCard.fillna(1, inplace=True)
# "god" samples
X.loc[X.bankCard > 1, 'bankCard'] = 0
# test data
testX.bankCard.replace(-999, 1, inplace=True)
testX.bankCard.fillna(1, inplace=True)
testX.loc[testX.bankCard > 1, 'bankCard'] = 0
X.duplicated().sum()


# 至此，我们将仅有的含有"缺失值"的列进行了处理,且数据没有重复数据。
# 注意这里仅仅处理了**空**的缺失值，每个变量内部还有另一种以`-999`填补好的缺失值。

# ### 可视化分析X.columns[:20]
X.columns[80:]
[col for col in X.columns if 'x_' not in col]
df = pd.concat([X, y.target], axis=1)


# #### 做图辅助函数def catPlot(feature, figsize=(14, 6), save=False, filename=None):
    feature_name = feature.capitalize()
    tmp = pd.crosstab(df[feature], df['target'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=figsize)
    plt.suptitle(f'{feature_name} Distributions', fontsize=22)

    plt.subplot(121)
    g = sns.countplot(x=feature, data=df)
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])

    g.set_title(f"{feature_name} Distribution", fontsize=19)
    g.set_xlabel(feature_name, fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 

    plt.subplot(122)
    g1 = sns.countplot(x=feature, hue='target', data=df)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=feature, y='Fraud', data=tmp, 
                       color='black', scale=0.5,
                       legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title(f"{feature_name} by Target(isFraud)", fontsize=19)
    g1.set_xlabel(feature_name, fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    plt.subplots_adjust(hspace = 0.6, top = 0.85)
    if save:
        if filename:
            plt.savefig(f'../pics/{filename}.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'../pics/{feature_name}.png', bbox_inches='tight', dpi=300)
    plt.show()
# numeric: box plot
def boxPlot(feature, save=False, filename=None):
    feature_name = feature.capitalize()
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
    axs[0].boxplot(df.loc[df.target == 0, feature])
    axs[0].set_xlabel('NoFraud')
    axs[1].boxplot(df.loc[df.target == 1, feature])
    axs[1].set_xlabel('Fraud')
    plt.suptitle(feature_name)
    if save:
        if filename:
            plt.savefig(f'../pics/{filename}.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'../pics/{feature_name}.png', bbox_inches='tight', dpi=300)
    plt.show()
# numeric: distribution plot
def distPlot(feature, save=False, filename=None):
    feature_name = feature.capitalize()
    g = sns.distplot(df[df['target'] == 1][feature], label='Fraud')
    g = sns.distplot(df[df['target'] == 0][feature], label='NoFraud')
    g.legend()
    g.set_title(f"{feature_name} Distribution by Target", fontsize=20)
    g.set_xlabel(feature_name, fontsize=18)
    g.set_ylabel("Probability", fontsize=18)
    if save:
        if filename:
            plt.savefig(f'../pics/{filename}.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'../pics/{feature_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


# #### 编码辅助函数def recode(feature, pre_name=None, df=df, testX=testX):
    temp = pd.get_dummies(pd.concat([df[feature], testX[feature]],axis=0), 
                              prefix=pre_name)
    df = df.join(temp.iloc[0:len(df), :])
    df.drop([feature], axis=1, inplace=True)
    # test data
    testX = testX.join(temp.iloc[len(df):, :])
    testX.drop([feature], axis=1, inplace=True)
    return df, testX


# #### 分类型变量
# `loanProduct`，`gender`等重新编码(one-hot)。大概分成P，N有明显差别和无明显差别（重叠）[保守策略，保留重叠特征，因为重叠特征较多]
# 
# 保留:`loanProduct`,`gender`,`job`,`ethnic`,`linkRela`,`setupHour`,
# `weekday`,`ncloseCreditCard`,`unpayIndvLoan`, `unpayOtherLoan`, 
# `unpayNormalLoan`, `5yearBadloan`,`basicLevel`
# 
# 去除：`id`， `certId`, `dist`，`residentAddr`,`isNew`

# ##### id
# testX的id要暂时保留，用于输出数据结果，但是在进入预测时要再去掉df.drop(['id'], axis=1, inplace=True)
# certId
df[df.target==0].certId.value_counts().head(10)


# ##### certId# 暂时去掉
df.drop(['certId'], axis=1, inplace=True)
testX.drop(['certId'], axis=1, inplace=True)


# ##### distpd.crosstab(df.dist, df.target).head(10)
df.drop(['dist'], axis=1, inplace=True)
testX.drop(['dist'], axis=1, inplace=True)


# ##### residentAddrpd.crosstab(df.residentAddr, df.target).head(10)
df.drop(['residentAddr'], axis=1, inplace=True)
testX.drop(['residentAddr'], axis=1, inplace=True)


# ##### isNew# isNew暂时删除
df.drop(['isNew'], axis=1, inplace=True)
testX.drop(['isNew'], axis=1, inplace=True)
df.head()


# ##### loanProductcatPlot('loanProduct', save=True)
df, testX = recode('loanProduct', pre_name = 'product')
df.head()


# ##### gender# gender
catPlot('gender', save=True)


# ##### job# job
catPlot('job', figsize=(18, 8), save=True)
# 共15种工作
job_names = ['job_' + str(i) for i in range(1, 16)]
# 编码
df[job_names] = pd.get_dummies(pd.concat([df.job, testX.job], axis=0)).iloc[0:len(df), :]
testX[job_names] = pd.get_dummies(pd.concat([df.job, testX.job], axis=0)).iloc[len(df):, :]
# 去除原始变量
df.drop(['job'], axis=1, inplace=True)
testX.drop(['job'], axis=1, inplace=True)


# ##### ethnicfig = plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
# 不显示汉族
sns.countplot(df.ethnic[(df.target == 0) & (df.ethnic != 0)])
plt.subplot(1, 2, 2)
sns.countplot(df.ethnic[(df.target == 1) & (df.ethnic != 0)])
plt.grid()
plt.show()
# 汉族为0，其他为1
df.loc[df.ethnic == 0, 'ethnic'] = 0
df.loc[df.ethnic != 0, 'ethnic'] = 1
testX.loc[testX.ethnic == 0, 'ethnic'] = 0
testX.loc[testX.ethnic != 0, 'ethnic'] = 1
# 去除原始变量
df.drop(['ethnic'], axis=1, inplace=True)
testX.drop(['ethnic'], axis=1, inplace=True)


# ##### linkRelacatPlot('linkRela', save=True)
relations = ['relation_' + str(i) for i in range(9)]
df[relations] = pd.get_dummies(pd.concat([df.linkRela, testX.linkRela], axis=0)).iloc[0:len(df), :]
testX[relations] = pd.get_dummies(pd.concat([df.linkRela, testX.linkRela], axis=0)).iloc[len(df):, :]
# drop
df.drop(['linkRela'], axis=1, inplace=True)
testX.drop(['linkRela'], axis=1, inplace=True)


# ##### setupHour# setupHour
catPlot('setupHour', figsize=(18, 7), save=True)


# 看右图时候注意：
# 
# 1.欺诈部分太小了，使得红色部分很少（但是有:-）
# 
# 2.关注点可以放在折线图的点在对应柱型图的位置,可以大致认为点高于柱型图表示欺诈率较高，如5点；反之越低，如20点。
# 
# 据此，将早晨9点到晚上9点视作“平稳期”，晚上9点到早晨9点视作“危险期”。df.loc[(df.setupHour >= 9) & (df.setupHour <= 21), 'setupHour'] = 0
df.loc[df.setupHour < 9, 'setupHour'] = 1
df.loc[df.setupHour > 21, 'setupHour'] = 1

testX.loc[(testX.setupHour >= 9) & (testX.setupHour <= 21), 'setupHour'] = 0
testX.loc[testX.setupHour < 9, 'setupHour'] = 1
testX.loc[testX.setupHour > 21, 'setupHour'] = 1

df.drop(['setupHour'], axis=1, inplace=True)
testX.drop(['setupHour'], axis=1, inplace=True)


# ##### weekdaycatPlot('weekday', save=True)
df.loc[df.weekday > 2, 'weekday'] = 0
df.loc[df.weekday <= 2, 'weekday'] = 1

testX.loc[testX.weekday > 2, 'weekday'] = 0
testX.loc[testX.weekday <= 2, 'weekday'] = 1

df.drop(['weekday'], axis=1, inplace=True)
testX.drop(['weekday'], axis=1, inplace=True)
df.head()


# ##### ncloseCreditCardcatPlot('ncloseCreditCard', save=True)
# recode(将-999缺失视作一种新的情况)
nclose = pd.get_dummies(pd.concat([df.ncloseCreditCard, testX.ncloseCreditCard],axis=0), 
                          prefix='nclose')
df = df.join(nclose.iloc[0:len(df), :])
df.drop(['ncloseCreditCard'], axis=1, inplace=True)
# test data
testX = testX.join(nclose.iloc[len(df):, :])
testX.drop(['ncloseCreditCard'], axis=1, inplace=True)


# ##### unpayIndvLoancatPlot('unpayIndvLoan')
# recode(将-999缺失视作一种新的情况)
unpay = pd.get_dummies(pd.concat([df.unpayIndvLoan, testX.unpayIndvLoan],axis=0), 
                          prefix='unpayInd')
df = df.join(unpay.iloc[0:len(df), :])
df.drop(['unpayIndvLoan'], axis=1, inplace=True)
# test data
testX = testX.join(unpay.iloc[len(df):, :])
testX.drop(['unpayIndvLoan'], axis=1, inplace=True)


# ##### unpayOtherLoancatPlot('unpayOtherLoan', save=True)
df, testX = recode('unpayOtherLoan', pre_name = 'unpayOther', df=df, testX=testX)
df.head()


# ##### unpayNormalLoancatPlot('unpayNormalLoan')
df, testX = recode('unpayNormalLoan', pre_name = 'unpayNormal', df=df, testX=testX)


# ##### 5yearBadloancatPlot('5yearBadloan')


# 无需额外编码。

# ##### basicLevelcatPlot('basicLevel', save=True)
df.basicLevel.replace([-999, 4], 1, inplace=True)
df.basicLevel.replace([1, 2, 3], 0, inplace=True)

testX.basicLevel.replace([-999, 4], 1, inplace=True)
testX.basicLevel.replace([1, 2, 3], 0, inplace=True)
df.head()


# #### 数值型变量
# > 由于某些分类数据（如教育`edu`）给出的值并非是类似1， 2， 3这种明显标识类别的值，而是类似40， 47，70这种，所以我们将其视为数值性数据，代表对应特征的程度。
# 
# `age`,`lmt`等去除异常值，进行标准化。
# 
# 全部保留：`age`,`edu`,`lmt`,`highestEdu`

# ##### ageboxPlot('age', save=True, filename='age[before]')
distPlot('age', save=True, filename='Age Dist Plot[before]')
df.drop(df.index[df.age > 100], inplace=True)


# 因为大于100岁的人属于多数类，且数量极少，对与我们的模型拟合没有正向作用，作为异常值去除。boxPlot('age', save=True, filename='age[after]')
distPlot('age', save=True, filename='Age Dist Plot[after]')
# df.drop(['age'], axis=1, inplace=True)
# # test data
# testX.drop(['age'], axis=1, inplace=True)


# 因为重叠过于严重，后续可以考虑去除。# 标准化
age_mean = df.age.mean()
age_std = df.age.std()
# training data
df.age = (df.age - age_mean) / age_std
# test data
testX.age = (testX.age - age_mean) / age_std


# 在标准化的时候注意要用训练集的统计量（均值，方差）对测试集的数据进行标准化，这是为了防止信息泄漏。
# 参考[stackoverflow1](https://stats.stackexchange.com/questions/327294/data-standardization-for-training-and-testing-sets-for-different-scenarios), [stackoverflow2](https://stats.stackexchange.com/questions/174823/how-to-apply-standardization-normalization-to-train-and-testset-if-prediction-i)

# ##### edu# edu
catPlot('edu')
df.drop(df.index[df.edu == -999], inplace=True)
df[df.target==0].edu.value_counts().head(10)
df[df.target==1].edu.value_counts().head(10)
# 标准化
edu_mean = df.edu.mean()
edu_std = df.edu.std()
# training data
df.edu = (df.edu - edu_mean) / edu_std
# test data
testX.edu = (testX.edu - edu_mean) / edu_std


# ##### lmt# lmt
boxPlot('lmt')
distPlot('lmt')
# 标准化
lmt_mean = df.lmt.mean()
lmt_std = df.lmt.std()
df.lmt = (df.lmt - lmt_mean) / lmt_std
testX.lmt = (testX.lmt - lmt_mean) / lmt_std


# ##### highestEdu# highestEdu
catPlot("highestEdu", save=True)
df.highestEdu.replace(-999, 0, inplace=True)
testX.highestEdu.replace(-999, 0, inplace=True)
# 标准化
highestEdu_mean = df.highestEdu.mean()
highestEdu_std = df.highestEdu.std()
# training data
df.highestEdu = (df.highestEdu - highestEdu_mean) / highestEdu_std
# test data
testX.highestEdu = (testX.highestEdu - highestEdu_mean) / highestEdu_std


# #### 隐私变量# x1-x78 PCA
# df.columns[3:82]

# pca = SparsePCA(n_components=1)
# pca.fit(df[df.columns[3:82]])

# pcaX = pca.transform(df[df.columns[3:82]])
# df['pcaX'] = pcaX
# df.drop(df.columns[3:82], axis=1, inplace=True)

# test data
# testX['pcaX'] = pca.transform(testX[testX.columns[4:83]])
# testX.drop(testX.columns[4:83], axis=1, inplace=True)
df.head()


# ## 数据保存y = df.target
X = df.drop(['target'], axis=1)
trainX.shape
X.shape
testX.shape
X.to_csv('../data/features/trainX.csv', index=None)
y.to_csv('../data/features/trainY.csv', index=None)
testX.to_csv('../data/features/testX.csv', index=None)
