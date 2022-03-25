#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import BorderlineSMOTE
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sea
import math as m
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction import text
import nltk as nlp

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA






import pandas as pd
from imblearn.over_sampling import ADASYN#
from imblearn.over_sampling import SMOTE#
from imblearn.over_sampling import KMeansSMOTE#
from imblearn.over_sampling import BorderlineSMOTE#
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import RandomOverSampler#
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek#
from imblearn.combine import SMOTEENN#

from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import TomekLinks


from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier 
from sklearn.linear_model import PassiveAggressiveClassifier 
from sklearn.tree import ExtraTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier 


from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 


from sklearn.model_selection import KFold 
from sklearn.model_selection import StratifiedKFold
import operator
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
import math as m
import math
import collections


# In[2]:


def Average(lst):
    return float(sum(lst) / len(lst)) 
kf = KFold(n_splits=5, random_state=None, shuffle=True)


# In[3]:


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


# In[4]:


#importing the datasets
rnaData = pd.read_csv("/Volumes/Backup Plus/ALL DOCS/Canada/UofAlberta CS/Courses/Winter/CMPUT690/project/GeneData/pancan_scaled_zeroone_rnaseq.tsv", sep='\t')
rnaData.rename(columns={"Unnamed: 0": "sample_id"}, inplace=True)
clinicalData = pd.read_csv("/Volumes/Backup Plus/ALL DOCS/Canada/UofAlberta CS/Courses/Winter/CMPUT690/project/GeneData/clinical_data.csv")
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


# In[5]:


#Undersampling

kf = StratifiedKFold(n_splits=4)

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

#warnings.filterwarnings('ignore')
f_sns = []
f_pre = []
f_f1 = []
f_spc = []
f_gmn =[]


#model_list = ['DTC','MLP','SVM','LRC','KNN','BNB','LDA','SGDC','PAC','ETC','AdaBoost','Bagging','GradientBoosting','RandomForest','DMLP']

model_list = ['DTC','MLP','SVM','LRC','KNN','BNB','LDA','SGDC','PAC','ETC','AdaBoost','Bagging','RandomForest','DMLP']
resampling_list = ['ClusterCentroids','CondensedNearestNeighbour','RandomUnderSampler','NeighbourhoodCleaningRule','EditedNearestNeighbours','AllKNN','RepeatedEditedNearestNeighbours','InstanceHardnessThreshold','NearMiss','OneSidedSelection','TomekLinks']

classifier = DecisionTreeClassifier()
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


    for resampler in resampling_list:
        f_sns = []
        f_pre = []
        f_f1 = []
        f_spc = []
        f_gmn =[]
        ln=[]
        print("Resampling Strategy: ",resampler)
        cnt=0
        for train_index, test_index in kf.split(X,y):
            sns = []
            pre = []
            f1 = []
            spc = []
            gmn =[]
            cnt+=1
            print('Fold: ',cnt)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            elements_count = collections.Counter(y_train)
            print("Class Distribution of Training Data: ", elements_count)
            all_classes = elements_count.keys()
            # printing the element and the frequency
            #dct = collections.Counter(y)
            majority_class=max(elements_count.items(), key=operator.itemgetter(1))[0]
            majority_count=elements_count[majority_class]
            strategy = {}
            ml_model = []
            resample_tech = []
            #for ratio in np.arange(0, 1.1, 0.1):
            strategy = {}
            #ml_model.append(model)
            #resample_tech.append(resampler)
            #print("Rat",ratio)
            #if ratio != 0:

            if resampler == 'ClusterCentroids':
                sm=ClusterCentroids()
              #sm=ADASYN(sampling_strategy=strategy)
            elif resampler == 'CondensedNearestNeighbour':
                sm=CondensedNearestNeighbour()
            elif resampler == 'RandomUnderSampler':
              #sm=KMeansSMOTE(sampling_strategy=strategy,kmeans_estimator=21)
                sm=RandomUnderSampler()
            elif resampler == 'NeighbourhoodCleaningRule':
                sm=NeighbourhoodCleaningRule()
            elif resampler == 'EditedNearestNeighbours':
                sm=EditedNearestNeighbours()
            elif resampler == 'AllKNN':
                sm=AllKNN()
            elif resampler == 'RepeatedEditedNearestNeighbours':
                sm=RepeatedEditedNearestNeighbours()
            elif resampler == 'InstanceHardnessThreshold':
                sm=InstanceHardnessThreshold()
            elif resampler == 'NearMiss':
                sm=NearMiss()
            elif resampler == 'OneSidedSelection':
                sm=OneSidedSelection()
            elif resampler == 'TomekLinks':
                sm=TomekLinks()




            print(y_train)
            try:
                x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
                print()
                #print("Class Distribution of Training Data after Oversampling at Ratio",ratio," : ", collections.Counter(y_train_res))
                classifier.fit(x_train_res,y_train_res)
                y_pred = classifier.predict(X_test)
                sns.append(sensitivity_score(y_test, y_pred, average='weighted'))
                spc.append(specificity_score(y_test, y_pred, average='weighted'))
                pre.append(precision_score(y_test, y_pred, average='weighted'))
                f1.append(f1_score(y_test, y_pred, average='weighted'))
                gmn.append(geometric_mean_score(y_test, y_pred, average='weighted'))
            except:
                continue

            #incr = ratio+0.1
            #print('incr: ',incr)
            #ratio = round_up(incr,2)
            #break
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
        #f_OQ = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
        ml_model = [model]
        resample_tech = [resampler]
        ml_model_df=pd.DataFrame(ml_model,columns=['Model'])
        resample_tech_df=pd.DataFrame(resample_tech,columns=['Resampling'])
        #f_OQ_df=pd.DataFrame(f_OQ,columns=['Oversampling Quantity'])
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
        if model == 'DTC' and resampler == 'ClusterCentroids':
            df_res = DTC_SMOTE_Res
        else:
            df_res = pd.concat([df_res,DTC_SMOTE_Res],axis=0)


# In[ ]:




