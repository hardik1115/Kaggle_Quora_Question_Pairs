#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf8 -*-

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn import preprocessing


h = .02  # step size in the mesh

names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest","ExtraTreesClassifier",  "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),

    #XGBClassifier(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    ExtraTreesClassifier(n_estimators=100, max_depth=5),
    
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

df = pd.read_csv('/Quora Question Pairs/Codes_Data/quora_features.csv')
import pandas as pd

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)




'''
df=pd.concat([df1, df2], axis=0)
df=pd.concat([df, df3], axis=0)
df=pd.concat([df, df4], axis=0)
'''
#"stats1","stats2","stats3","stats4","hurst","pfd","hjorth","katz","higuchi","shannon_entropy","skewness","kurtosis","psd1","psd2","psd3","psd4","psd5","ev1","ev2","ev3","ev4","ev5","ev6","ev7","ev8","ev9","ev10","ev11","ev12","ev13","ev14","ev15","Class"

Z= df.drop(['question1','question2'], 1)
Z=clean_dataset(Z)

y = Z['is_duplicate']

X = Z.drop(['is_duplicate'], 1)



min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X.dropna())


print X.head()
# iterate over datasets

    # preprocess dataset, split into training and test part

#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=Z.is_duplicate,random_state=42)




import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')










# iterate over classifiers
for name, clf in zip(names, classifiers):
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print name+' Score:',score
        prediction1 = clf.predict(X_test)



        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test,prediction1)
        np.set_printoptions(precision=2)
        
        print name+" Recall metric in the testing dataset: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1]))
        
        # Plot non-normalized confusion matrix
        class_names = [0,1]
        plt.figure()
        plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
        plt.savefig('/home/lab3/Documents/HB/Quora_Ques/Plots/'+name+'_confusion_matrix.png')
        

        print name+" Precision: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))

        recall=np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1]))
        print name+" Recall: ", recall
        precision=np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))
        f1score=2*np.true_divide(precision*recall,(precision+recall))

        print name+" F1 Score: ", f1score
        







