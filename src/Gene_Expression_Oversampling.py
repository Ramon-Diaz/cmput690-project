#!/usr/bin/env python
# coding: utf-8
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import precision_score, f1_score, precision_score

from imblearn.over_sampling import SVMSMOTE, ADASYN, SMOTE, KMeansSMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN

from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import operator
import collections
import pathlib
import time

import multiprocessing as mp
# %%
def Average(lst):
    return float(sum(lst) / len(lst)) 
kf = KFold(n_splits=5, random_state=None, shuffle=True)

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# %%
def import_data():
    print('Importing the data...')
    #importing the datasets
    rnaData = pd.read_csv(pathlib.Path.cwd().joinpath('data/pancan_scaled_zeroone_rnaseq.tsv'), sep='\t')
    rnaData.rename(columns={"Unnamed: 0": "sample_id"}, inplace=True)
    clinicalData = pd.read_csv(pathlib.Path.cwd().joinpath('data/clinical_data.csv'))
    rnaData_columns = rnaData.columns
    clinicalData_columns = clinicalData.columns
    #merging clinical data and rnadata
    combined = pd.merge(rnaData, clinicalData, on="sample_id")
    #dropping the irrelevant columns
    combined.drop(['sample_id', 'days_to_death', 'platform', 'analysis_center', 'gender',
        'race', 'ethnicity', 'organ', 'vital_status',
        'sample_type', 'age_at_diagnosis', 'percent_tumor_nuclei', 'drug',
        'year_of_diagnosis'],axis =1, inplace=True)
    #dropping null values
    combined.dropna(inplace=True)


    #plotting the class distribution
    class_dist=combined['acronym'].value_counts()
    classes = np.array(class_dist.index)
    class_count = np.array(class_dist.values)

    return combined, classes, class_count

def plot(classes, class_count):

    fig, ax = plt.subplots(figsize =(16, 9))
    ax.barh(classes, class_count)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    
    # Show top values
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
    # Add Plot Title
    ax.set_title('Class Distribution in the Dataset',
                loc ='left', )
    
    # Show Plot
    plt.show()


def encode_reduce(combined):
    print('Encoding with PCA...')
    labelencoder = LabelEncoder()
    X=np.array(combined.iloc[:,1:-3])
    y=labelencoder.fit_transform(np.array(combined.iloc[:,-3]))
    y_stage = labelencoder.fit_transform(np.array(combined.iloc[:,-1]))


    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA()
    pca.fit_transform(X)
    total = sum(pca.explained_variance_)
    k = 0
    current_variance = 0
    while current_variance/total < 0.90:
        current_variance += pca.explained_variance_[k]
        k = k + 1

    print(k, " features explain around 90% of the variance. From 7129 features to ", k, ", not too bad.", sep='')
    pca = PCA(n_components=k)
    pca.fit(X)
    X = pca.transform(X)

    dct = collections.Counter(y)
    key=dct.keys()

    return X, y

def run_sampling_technique(df_res, X, y, kf, classifier, resampler):
    f_sns = []
    f_pre = []
    f_f1 = []
    f_spc = []
    f_gmn =[]
    ln=[]
    #print("Resampling Strategy: ",resampler)
    cnt=0
    for train_index, test_index in kf.split(X,y):
        sns = []
        pre = []
        f1 = []
        spc = []
        gmn =[]
        cnt+=1
        print('    Resampling Strategy: {0}, Fold: {1}'.format(resampler,cnt))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        elements_count = collections.Counter(y_train)
        #print("Class Distribution of Training Data: ", elements_count)
        all_classes = elements_count.keys()
        # printing the element and the frequency

        majority_class=max(elements_count.items(), key=operator.itemgetter(1))[0]
        majority_count=elements_count[majority_class]
        strategy = {}
        ml_model = []
        resample_tech = []
        for ratio in np.arange(0, 1.1, 0.1):
            strategy = {}
    
            if ratio != 0:
                for class_label in all_classes:
                    if class_label != majority_class:
                        count_current = elements_count[class_label]
                        maj_min_diff = majority_count - count_current
                        syntheticData_amount=(float(ratio*100)/100)*maj_min_diff
                        syntheticData_amount=int(math.ceil(syntheticData_amount))
                        strategy[class_label] = syntheticData_amount + elements_count[class_label]
                    else:
                        strategy[class_label] = majority_count

                if resampler == 'ADASYN':
                    sm=ADASYN(n_neighbors=30,sampling_strategy=strategy)
                    #sm=ADASYN(sampling_strategy=strategy)
                elif resampler == 'SMOTE':
                    sm=SMOTE(sampling_strategy=strategy)
                elif resampler == 'KMeansSMOTE':
                #sm=KMeansSMOTE(sampling_strategy=strategy,kmeans_estimator=21)
                    sm=KMeansSMOTE(sampling_strategy=strategy)
                elif resampler == 'BorderlineSMOTE-1':
                    sm=BorderlineSMOTE(sampling_strategy=strategy,kind="borderline-1")
                elif resampler == 'BorderlineSMOTE-2':
                    sm=BorderlineSMOTE(sampling_strategy=strategy,kind="borderline-1")
                elif resampler == 'SVMSMOTE':
                    sm=SVMSMOTE(sampling_strategy=strategy)
                elif resampler == 'RandomOversampler':
                    sm=RandomOverSampler(sampling_strategy=strategy)
                elif resampler == 'SMOTENC':
                    sm=SMOTENC(sampling_strategy=strategy)
                elif resampler == 'SMOTETomek':
                    sm=SMOTETomek(sampling_strategy=strategy)
                elif resampler == 'SMOTEENN':
                    sm=SMOTEENN(sampling_strategy=strategy)

                try:
                    x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
                    #print()
                    #print("Class Distribution of Training Data after Oversampling at Ratio",ratio," : ", collections.Counter(y_train_res))
                    classifier.fit(x_train_res,y_train_res)
                    y_pred = classifier.predict(X_test)
                    sns.append(sensitivity_score(y_test, y_pred, average='weighted'))
                    spc.append(specificity_score(y_test, y_pred, average='weighted'))
                    pre.append(precision_score(y_test, y_pred, average='weighted'))
                    f1.append(f1_score(y_test, y_pred, average='weighted'))
                    gmn.append(geometric_mean_score(y_test, y_pred, average='weighted'))
                except:
                    print("Not working!!")
                    continue
            else:
                classifier.fit(X_train,y_train)
                y_pred = classifier.predict(X_test)
                sns.append(sensitivity_score(y_test, y_pred, average='weighted'))
                spc.append(specificity_score(y_test, y_pred, average='weighted'))
                pre.append(precision_score(y_test, y_pred, average='weighted'))
                f1.append(f1_score(y_test, y_pred, average='weighted'))
                gmn.append(geometric_mean_score(y_test, y_pred, average='weighted'))

        f_sns.append(sns)
        f_pre.append(pre)
        f_f1.append(f1)
        f_spc.append(spc)
        f_gmn.append(gmn)

    print(np.mean(f_sns, axis = 0))
    print(np.mean(f_pre, axis = 0))
    print(np.mean(f_f1, axis = 0))
    print(np.mean(f_spc, axis = 0))
    print(np.mean(f_gmn, axis = 0))
    f_OQ = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
    ml_model = [model,model,model,model,model,model,model,model,model,model,model]
    resample_tech = [resampler,resampler,resampler,resampler,resampler,resampler,resampler,resampler,resampler,resampler,resampler]
    ml_model_df=pd.DataFrame(ml_model,columns=['Model'])
    resample_tech_df=pd.DataFrame(resample_tech,columns=['Resampling'])
    f_OQ_df=pd.DataFrame(f_OQ,columns=['Oversampling Quantity'])
    f_sns_df=pd.DataFrame(np.mean(f_sns, axis = 0),columns=['SNS'])
    f_sns_std_df=pd.DataFrame(np.std(f_sns, axis = 0),columns=['SNS STD'])
    f_pre_df=pd.DataFrame(np.mean(f_pre, axis = 0),columns=['PRE'])
    f_pre_std_df=pd.DataFrame(np.std(f_pre, axis = 0),columns=['PRE STD'])
    f_f1_df=pd.DataFrame(np.mean(f_f1, axis = 0),columns=['F1'])
    f_f1_std_df=pd.DataFrame(np.std(f_f1, axis = 0),columns=['F1 STD'])
    f_spc_df=pd.DataFrame(np.mean(f_spc, axis = 0),columns=['SPC'])
    f_spc_std_df=pd.DataFrame(np.std(f_spc, axis = 0),columns=['SPC STD'])
    f_gmn_df=pd.DataFrame(np.mean(f_gmn, axis = 0),columns=['GMN'])
    f_gmn_std_df=pd.DataFrame(np.std(f_gmn, axis = 0),columns=['GMN STD'])
    DTC_SMOTE_Res=pd.concat([ml_model_df,resample_tech_df,f_OQ_df,f_sns_df,f_sns_std_df,f_pre_df,f_pre_std_df,f_f1_df,f_f1_std_df,f_spc_df,f_spc_std_df,f_gmn_df,f_gmn_std_df],axis = 1)
    if model == 'DTC' and resampler == 'ADASYN':
        df_res = DTC_SMOTE_Res
        print(df_res)
    else:
        df_res = pd.concat([df_res,DTC_SMOTE_Res],axis=0)
        print(df_res)

    return df_res

def get_result(result):
    global results
    results.append(result)

# %%
if __name__ == '__main__':
    combined, classes, class_count = import_data()
    #plot(classes, class_count)
    X, y = encode_reduce(combined)

    ts = time.time()
    cnt=0
    df_res = pd.DataFrame()

    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    #model_list = ['DTC','MLP','SVM','LRC','KNN','BNB','LDA','SGDC','PAC','ETC','AdaBoost','Bagging','GradientBoosting','RandomForest','DMLP']
    model_list = ['DTC','KNN']

    #resampling_list = ['SMOTE','BorderlineSMOTE-1','BorderlineSMOTE-2','SVMSMOTE','RandomOversampler','SMOTETomek','SMOTEENN']
    resampling_list = ['SMOTE','BorderlineSMOTE-1']

    ln=[]

    for model in model_list:

        if model == 'DTC':
            classifier = DecisionTreeClassifier()
        elif model == 'MLP':
            classifier = MLPClassifier()
        elif model == 'SVM':
            classifier = LinearSVC()
        elif model == 'LRC':
            classifier = LogisticRegression()
        elif model == 'KNN':
            classifier = KNeighborsClassifier()
        elif model == 'BNB':
            classifier = BernoulliNB()
        elif model == 'LDA':
            classifier = LinearDiscriminantAnalysis()
        elif model == 'SGDC':
            classifier = SGDClassifier()
        elif model == 'PAC':
            classifier = PassiveAggressiveClassifier()
        elif model == 'ETC':
            classifier = ExtraTreeClassifier()
        elif model == 'AdaBoost':
            classifier = AdaBoostClassifier()
        elif model == 'Bagging':
            classifier = BaggingClassifier()
        elif model == 'GradientBoosting':
            classifier = GradientBoostingClassifier()
        elif model == 'RandomForest':
            classifier = RandomForestClassifier()
        elif model == 'DMLP':
            classifier = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=200)
        print("Model: ",model)

        results = []
        pool = mp.Pool(2)
        for resampler in resampling_list:
            pool.apply_async(run_sampling_technique, args=(df_res, X, y, kf, classifier, resampler) , callback=get_result)
        pool.close()
        pool.join()

        print(results)
    print('Time in serial:', time.time() - ts)