#!/usr/bin/python

import sys
import pickle
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



def Draw(pred, features, poi, mark_poi=False, 
    name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ Drawing scatter plots with the features and storing in png """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(poi):
        #plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])
        plt.scatter(features[ii][0], features[ii][1], color = "b")
    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(poi):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    import math
    
    if ((~math.isnan(float(poi_messages))) &
         (~math.isnan(float(all_messages))) & (all_messages != 0)):
    #if (poi_messages != 'NaN' & all_messages != 'NaN' & float(all_messages) != 0.):
        return float(poi_messages)/float(all_messages)
    return 0


def calculateMetrics(clf, features_test, labels_test):

    pred = clf.predict(features_test)
    print "accuracy: ",clf.score(features_test, labels_test)
    print "f1-score: ",metrics.f1_score(labels_test, pred)  
    print "precision: ",metrics.precision_score(labels_test, pred)
    print "recall: ",metrics.recall_score(labels_test, pred)

def viewNaN(dict):
    """Visualize the number of NaN present in each of the features of the dict"""
    from collections import defaultdict

    features_dict = defaultdict(int)
    for key, elem in dict.iteritems():
        cont = 0
        for feat, value in elem.iteritems():
            if value == 'NaN':
                cont += 1
                features_dict[feat] +=1
            elif value == 0 or value == 0.0:
                cont += 1
        if cont > 19:
            print "\nEmpty Registry:", key
        #if elem['long_term_incentive']!= "NaN" :
        #    print "\nmax:", key, "elem:",elem['poi'], " - ", elem['long_term_incentive']
        #if elem['from_this_person_to_poi'] != "NaN" and float(elem['from_this_person_to_poi'])> 400:
            #print "max:", key, "elem:",elem
        #if elem['shared_receipt_with_poi'] != "NaN" and float(elem['shared_receipt_with_poi']) < 100 and elem['poi'] == True and float(elem['salary']) > 400000:
        if elem['salary'] != "NaN" and float(elem['salary']) > 25000000 :
            print "\nPossible outlier:", key, ":",elem 

    
    features_df = pd.DataFrame.from_dict(features_dict.items())
    features_df.columns = ['Feature', 'NaN']
    try:
        sorted_df = features_df.sort_values(by='NaN', ascending=True)
        print "\nNumber of NaN values in dictionary:"
        print sorted_df
    except AttributeError:
        print "\nNumber of NaN values in dictionary (unordered due to the pandas version used):"
        print features_df
    

def exploreData(dict, name=None):
    """ Prints the basic measures of data"""
    print "\nEXPLORING DATA"
    print "People in the dataset:", len(dict.keys())
    if name:
        print "Features + label in the dataset:", len(data_dict[name].keys()), ":", data_dict[name].keys()


def addFeature(dict, features_list):
    """Adding new features to the dictionary"""
    print "\nADDING NEW FEATURES ..."
    for name in dict:

        data_point = data_dict[name]
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_point["fraction_to_poi"] = fraction_to_poi
    #print "fraction_from_poi:", fraction_from_poi
    #print "fraction_to_poi:", fraction_to_poi
    features_list.append("fraction_from_poi")
    features_list.append("fraction_to_poi")
    return dict, features_list

def featureScaling(features):
    """Scaling the provided features, and returning the new scaled features """
    print "\nSCALING ..."
    scaler = MinMaxScaler()
    rescaled_features = scaler.fit_transform(features)
    #print "rescaled_features:", rescaled_features
    return rescaled_features

def featureSelection(features, labels, features_list, k=5, function=f_classif):
    """Selecting a k number of features with a diven function, it will print the 
    scoring calculated with three different scoring functions"""
    print "\nCHECKING FEATURES SCORES..."
    if function == "all":
        selector_f = SelectKBest(f_classif, 5)
        selected_features_F = selector_f.fit_transform(features, labels)
        selector_chi = SelectKBest(chi2, 5)
        selected_features_chi = selector_chi.fit_transform(features, labels)
        selector_mutual = SelectKBest(mutual_info_classif, 5)
        selected_features_mutual = selector_mutual.fit_transform(features, labels)

        features_scores_df = pd.DataFrame(
        {'feature': features_list[1:],
         'score_f_classif': selector_f.scores_,
         'score_chi2': selector_chi.scores_,
         'score_mutual': selector_mutual.scores_
        })

        try:
            sorted_df = features_scores_df.sort_values(by='score_f_classif', ascending=False)
            print "Score of the features (ordered by f_classif):"
            print sorted_df
        except AttributeError:

            print "Score of the features (unordered due to the pandas version used):"
            print features_scores_df
    else:
        ### Calculating the selected features according to the parameters
        selector = SelectKBest(function, k)
        selected_features = selector.fit_transform(features, labels)
        print "score function:",function
        features_scores_df = pd.DataFrame(
            {'feature': features_list[1:],
             'score': selector.scores_,
             })
        try:
            sorted_df = features_scores_df.sort_values(by='score', ascending=False)
            print "Scores of the %d features used (ordered):" %(k)
            print sorted_df[:k]
        except AttributeError:
            print "Scores of all the features (unordered due to the pandas version used):"
            print features_scores_df


def getValidationData(features, labels, test_size=0.1, n_iter=10, random_state=42):
    sss = StratifiedShuffleSplit(
        labels,
        test_size=test_size,
        n_iter=10,
        random_state=42
    )
    for train_idx, test_idx in sss: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    return features_train, features_test, labels_train, labels_test

def getValidationKfold(features, labels):
    kf = KFold(shuffle=True)
    for train_idx, test_idx in kf.split(features): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    return features_train, features_test, labels_train, labels_test


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 
    'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
    'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 
    'from_messages', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 
    'long_term_incentive', 'from_poi_to_this_person']

print "\n=================== STARTING PROCESS ========================"
print "\nInitial feature list we are going to use:", features_list


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



# DATA EXPLORATION
exploreData(data_dict, name='LAY KENNETH L')
viewNaN(data_dict)

print "\nLength of initial features list:",len(features_list)-1


### Task 2: Remove outliers
print "\nREMOVING OUTLIERS..."
data_dict.pop('TOTAL')
exploreData(data_dict, name='LAY KENNETH L')

### Task 3: Create new feature(s)
data_dict, features_list = addFeature(data_dict, features_list)
print "Length of features list:", len(features_list)-1

### Store to my_dataset for easy export below.
my_dataset = data_dict
print "\nData set length:", len(my_dataset.keys())

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
print "\nFeatures array length:", len(data)
labels, features = targetFeatureSplit(data)

# FEATURE VISUALIZATION
#Draw(None, features, labels, mark_poi=True, name="TEST.pdf", f1_name=features_list[1], f2_name=features_list[2])


# FEATURE SCALING
features_scaled = featureScaling(features)

#FEATURE SELECTION
# Checking the scores of the features
featureSelection(features_scaled, labels, features_list, function="all")



### WE REPEAT TASK 3 WITH A NEW FEATURES_LIST
#print "\n\n============ RESTARTING PROCESS WITH NEW FEATURES LIST =============" 
#print "============ APPLYING VALUES CALCULATED WITH GRIDSEARCHCV =========="
#print "\n\nDELETING DISCARDED FEATURES FROM FEATURES LIST"
#features_list = ['poi', 'salary', 'to_messages', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']
#print "Length of features list:",len(features_list)-1
#data_dict, features_list = addFeature(data_dict, features_list)

### Store to my_dataset for easy export below.
#my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
#print "Dataset Features we are going to use:",len(features[0])

# # FEATURE SCALING
#features_scaled = featureScaling(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# This is done with GridSearchCV later


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

tuning = False

if tuning:
    # PIPELINE AND GRIDSEARCH 


    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest()),
        ('dimens_red', PCA()),
        ('classify', GaussianNB())
    ])


    param_grid = [
        {
            'selector__score_func': [f_classif, mutual_info_classif, chi2],
            'selector__k': range(4, 9),
            'dimens_red__n_components': [2, 3, 4],
            'classify': [GaussianNB()]
        },
        {
            'selector__score_func': [f_classif, mutual_info_classif, chi2],
            'selector__k': range(4, 9),
            'dimens_red__n_components': [2, 3, 4],
            'classify': [ SVC()],
            'classify__kernel': ['linear', 'poly', 'rbf'],
            'classify__C': [1, 10, 100, 1000]
        } ,
        {
            'selector__score_func': [f_classif, mutual_info_classif, chi2],
            'selector__k': range(4, 9),
            'dimens_red__n_components': [2, 3, 4],
            'classify': [DecisionTreeClassifier()],
            'classify__min_samples_split': range(2, 5),
            'classify__max_features': ['auto', None]
        }
    ]


    cv = StratifiedShuffleSplit(
            labels,
            test_size=0.1,
            n_iter=10,
            random_state=42
        )



    clf_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1')
    clf_search.fit(features,labels)


    print "\nBEST ESTIMATOR",clf_search.best_estimator_
    print "\n\nBEST PARAMS:",clf_search.best_params_
    print "\nBEST CLASSIFIER:", clf_search.best_params_['classify']
    print "\nBEST_SCORE",clf_search.best_score_

    print "\nFEATURES SCORES:"

    featureSelection(
        features_scaled,
        labels,
        features_list,
        k=clf_search.best_params_['selector__k'],
        function=clf_search.best_params_['selector__score_func']
        )

    clf = clf_search.best_estimator_

else:
    print "\nUSING THE BEST CLASSIFIER CALCULATED BY GRIDSEARCHCV"
    clf = Pipeline([
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(k=5, score_func=chi2)),
        ('dimens_red', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)),
        ('classify', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'))
    ])

    featureSelection(
        features_scaled,
        labels,
        features_list,
        k=5,
        function=chi2
        )


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)

