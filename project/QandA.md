
### QUESTIONS AND ANSWERS

####1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]


Our dataset contains both financial and email information from people related with the ENRON scandal. After ENRON bankruptcy in 2001, it was proved that for years, the company committed massive fraud to hide its debts while pretending to be highly profitable. Many of the executives and directors of the firm were prosecuted. With the current dataset we want to answer if it is possible to predict who was a POI (person of interest: indicted, settled without admitting guilt or testified in exchange for immunity) based on their financial and mailing information. In our EDA (Exploratory Data Analysis) we discover that we have data from 146 people, 18 of which are POIs. They are already classified by the label 'poi', therefore this is a case where we can use Machine Learning Supervised Classification techniques. Still, these data is far from ideal, due to its scarcity. It covers just a small group of people, with a really small number of positively labeled POIs, hence the decision about how we divide data for training, validation and testing is critical. A wrong division could lead us to very inaccurate results. We'll discuss this point later.

During the exploration of the dataset we realize that one of the points is completely empty (LOCKHART EUGENE E), whose values for all the features are NaN, it will be discarded when formating the features, therefore, having our final dataset 145 people. We have 20 features for every person in the dataset, but not all of them contain information, we study the number of NaN (Not A Number) that we have for every feature in the dataset, and find that some of them have much more NaN than real values. An important issue now is searching for outliers, to find them, we create a Draw function to plot several scatter plots of the intuitively most interesting features. In plot outliers.pdf we can see that we have a value for TOTAL which definitely is a mistake from reading data out of a spreadsheet and we need to drop it. After deleting it we create more scatter plots and find other extreme points (see feat_x_x.pdf files), but searching for that names we see that they represent existing people with important roles in the company so they are real values that we want to consider for our study. From this graphical representation we can extract surprising cases, of non-POIs with outstanding values, like the bonus of LAVORATO JOHN, the salary of FREVERT MARK A, and the restricted options of WHITE JR THOMAS, a former US Secretary of Army. These names could be susceptible to a further journalistic investigation and some of them were in fact, judging by the information extracted from a brief Internet research. My point is how visual inspection and graphical representation can result in finding controversial/interesting cases for an investigation.

####2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

First, we need to use some intuition here. We discard the features 'other' and 'email_address' as they are not going to provide useful information for our goal (18 features now), then we create two new features (20 features), the ratio of messages received from a POI from all the received messages (fraction_from_poi), and the ratio of messages sent to a POI from all the messages sent by a person (fraction_to_poi). These features seem interesting since they could represent how much relation a person had with a POI and so, they could be a good way of identifying other POIs. After creating the new features, we have a look at the feature ranges, seeing that the financial features have much more high ranges of values than the email features (which is easy to see in the previous scatter plots). To avoid having unbalanced weights in these kind of features we should scale all of them using the MinMaxScaler function. We are doing this because we are considering several possible classifiers and dimensionality reduction algorithms, and while Decision Tree Classifier, for example, is scale invariant, Support Vector Machine and Principal Component Analysis (we'll talk about it later) are not, so it is highly recommended that we scale our data.

Finally we apply SelectKBest function to know the score of each feature when using it for predicting POIs. We compare three different score functions (chi2, f_classif, and mutual_info_classif) and see that the values differ depending on the function we used (see code output), so we decide not to discard features attending to its scores and test the score functions using GridSearchCV, to get the optimal score function and the optimal number of features (k) we are going to use in our classifier. Still, given the small size of the dataset, using a high number of features could lead us to overfitting (we will discuss overfitting later), also, tuning the function SelectKBest with the total range of possible features suppose an extremely high computational load, so we decide to tune the function searching for the optimal k in the range between 4 and 8 (range(4,9)).

####3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

Before the classifier, we applied dimensionality reduction (PCA), using several values for the n_component parameter in order to find the most optimal one with the help of GridSearchCV. Given that the maximum number of features we configured were in the range [4, 8], the tested values for the n_component parameter are [2, 3, 4], as the maximum number of components we can configure is the number of features we are using. Then we tried three different classifiers: Naive Bayes, SVM and Decision Tree. In order to pick up the most accurate according to f1-score that maximizes precision and recall together. To do so we used the cross validation tool GridSearchCV with a variety of options for the selector, the dimensionality reduction and the classifier. We also used a cross validation function based in StratifiedShuffleSplit(*) to get a training set of 90% of the data and a validation set of 10%, with 10 iterations. We fit the GridSearchCV to train and validate with the whole set of data, since, as we said in the first chapter, the amount of data is so small that separating a part out of it for testing would mean having a too small data for training and validating.

The highest F1-score resulted to be achieved by a Decision Tree Classifier with a specified set of parameters. I rerun GridSearchCV with a divided param_grid to compare with the performance of the other two classifiers. And these are the f1-scores obtained by the three of them:

|classifier    |f1-score| parameters                                                                          |
|--------------|--------|---------------------------------------------------------------------------------|
|Decision Tree |0.40    |score_func: chi2, k: 5, n_components: 2, min_samples_split: 2, max_features: auto|
|Naive Bayes   |0.30    |score_func: mutual_info_classif, k: 4, n_components: 3                           |
|SVM           |0.27    |score_func: mutual_info_classif, k: 4, n_components: 3, kernel: rbf, C: 1000     |

Therefore we can answer now, which features we ended up using based on the parameter optimization made by GridSearchCV for SelectKBest. The optimal number of features found is 5, and the list of the ones selected, ordered by its score is shown. We can see that one of our added features (fraction_to_poi) was included in this group.

|feature                |score|
|-----------------------|-----|
|exercised_stock_options| 6.93|
|loan_advances          | 6.74|
|total_stock_value      | 5.54|
|bonus                  | 5.19|
|fraction_to_poi        | 4.72|



(*) I used sklearn.cross_validation.StratifiedShuffleSplit, although it is deprecated in favor of sklearn.model_selection.StratifiedShuffleSplit, because the former is the one used in the testing script.


####4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

To tune the algorithms we used GridSearchCV with different sets of parameters for each kind of possible classifier used. The tuning of the parameters is important because different parameter sets mean different performance of the algorithms. GridSearchCV allows us to prove different combinations of steps, and we used combinations of feature selection parameters, as well as dimensionality reduction parameters and classifier parameters. We created a really loaded grid of parameters, so it takes a while for GridSearchCV to figure out the best combination, as long as time is not a problem we are being sure of selecting the right combination of parameters.

The parameters used to tune the classifier are the following:

|step                  |parameter                                                      |
|----------------------|---------------------------------------------------------------|
|SelectKBest           |k=range(4, 9), score_func=[f_classif, mutual_info_classif, chi2] |
|PCA                   |n_components=[2, 3, 4]                                         |
|GaussianNB            |                                                               |
|SVC()                 | kernel=['linear', 'poly', 'rbf'], C=[1, 10, 100, 1000]        |
|DecisionTreeClassifier|min_samples_split=range(2, 5), max_features'=['auto', None]    |



####5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Validation is the process in which we evaluate the capacity of our model to predict the target we are working with, in our case the probability of our algorithm to predict a POI based on the financial and email data we have in our data set. We do not use the same data to train our algorithm and to validate it. This way we protect ourselves from overfitting, which means creating a model that does not predict but replicate the results, having a very bad response to new data. By validating in different data we are sure that the model is predicting with unknown data, not replicating results.

The problem that we find in this case is that the dataset is really small, so dividing the set on a fix number of training and validating points can result in accuracy results, highly dependent on they way we split the data. Therefore, the correct way of validating in these cases is running a number of experiments in which we split the data differently, and finally calculate the average of the results. Furthermore, giving the small amount of positives that we have in our dataset, we should split the data trying to maintain the same proportion of positives and negatives in both the training and the validating data, we achieve this with the function StratifiedShuffleSplit.

####6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The metrics we are calculating are the following:

|metric   |value|
|---------|-----|
|accuracy |0.82 |
|f1-score |0.33 |
|precision|0.34 |
|recall   |0.33 |

Being the accuracy the ratio of correctly labeled points and the total points we are predicting and the f1-score a combination of precision and recall which are interesting metrics for our case. *Prediction* is the probability of a point to be a POI once it was predicted as a POI, and *Recall* is the probability of a real POI to be predicted as a POI. So, depending on how conservative we want to be with the prediction, if we want to minimize the false positives (considering somebody a POI when he/she was not) we should maximize the Precision, but, on the other hand if we prefer to minimize the false negatives (considering somebody a non-POI when he/she is a fact a POI) we should try to maximize the Recall. 




>I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

