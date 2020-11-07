import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.inspection import permutation_importance


data = pd.read_csv("raw_with_cluster_labels.csv")
lb_make = LabelEncoder()
data["Race_code"] = lb_make.fit_transform(data["Race"])
races = data.loc[:, ['Race']].values
data = data.drop("Race", axis=1)
y = data['GameWon']

data[data.columns] = preprocessing.MinMaxScaler().fit_transform(data.values)

# drop segment for testing
print("Segment column dropped")
data = data.drop('Segment', axis=1)

# 'Unnamed: 0' is the order number in the file, can be safely ignored
X_raw = data.drop(['Unnamed: 0', 'GameWon', 'Touchdowns', 'TouchdownsAgainst', 'Score', 'ScoreAgainst'], axis=1)

print("Entries in the dataset: ", len(data))
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.6)



def support_vector_classification():
	print("Support Vector Classifier")
	print("Training set size: ", len(X_train))
	classifier = SVC(kernel='linear')
	print("Fitting training set...")
	classifier.fit(X_train, y_train)

	print("Predicting...")
	y_predict = classifier.predict(X_test)
	print(confusion_matrix(y_test, y_predict))
	print(classification_report(y_test, y_predict))

	cross_validation(classifier)

	feature_importance = abs(classifier.coef_[0])
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5
	featfig = plt.figure()
	featax = featfig.add_subplot(1, 1, 1)
	featax.barh(pos, feature_importance[sorted_idx], align='center')
	featax.set_yticks(pos)
	featax.set_yticklabels(np.array(X_raw.columns)[sorted_idx], fontsize=8)
	# print(np.array(X_raw.columns)[sorted_idx])
	# print(feature_importance[sorted_idx])
	results = pd.DataFrame({'Importance': feature_importance[sorted_idx], "Feature": np.array(X_raw.columns)[sorted_idx]})
	print(results.sort_values("Importance"))

	featax.set_xlabel('Relative Feature Importance')
	plt.tight_layout()
	plt.show()


def cross_validation(classifier):
	pipeline = make_pipeline(StandardScaler(), classifier)
	#
	# Pass instance of pipeline and training and test data set
	# cv=10 represents the StratifiedKFold with 10 folds
	#
	scores = cross_val_score(pipeline, X=X_train, y=y_train, cv=10, n_jobs=1)

	print('Cross Validation accuracy scores: %s' % scores)

	print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def naive_bayes_classification():
	# dataset with cluster labels:
	# False - 82%
	# True - 73%
	#
	# [[57210 13337]
	#  [12976 36951]]

	# dataset without cluster labels:
	# False - 80%
	# True - 71%
	#
	# [[56482 14151]
	#  [13736 36105]]

	print("Naive Bayes Classifier")
	print("Training set size: ", len(X_train))
	nb = GaussianNB()
	print("Fitting training set...")
	nb.fit(X_train, y_train)
	print("Predicting...")
	y_predict = nb.predict(X_test)

	print(confusion_matrix(y_test, y_predict))
	print(classification_report(y_test, y_predict))

	cross_validation(nb)

	imps = permutation_importance(nb, X_test, y_test)
	feature_importance = abs(imps.importances_mean)
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	results = pd.DataFrame({'Permutation importance': feature_importance, "Feature": X_raw.columns.values})
	print(results.sort_values("Permutation importance"))


def logistic_regression_classification():
	# dataset with cluster labels:
	# False - 87%
	# True - 83%
	#
	# [[62490  8002]
	#  [9750 40232]]
	#
	# dataset without cluster labels: FAILED TO CONVERGE
	# False - 90%
	# True - 89%
	#
	# [[65212  5482]
	#  [7152 42628]]

	print("Logistic Regression Classifier")
	print("Training set size: ", len(X_train))
	classifier = LogisticRegression(max_iter=1000)
	print("Fitting training set...")
	classifier.fit(X_train, y_train)

	print("Predicting...")
	y_predict = classifier.predict(X_test)

	print(confusion_matrix(y_test, y_predict))
	print(classification_report(y_test, y_predict))

	cross_validation(classifier)

	feature_importance = abs(classifier.coef_[0])
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5
	featfig = plt.figure()
	featax = featfig.add_subplot(1, 1, 1)
	featax.barh(pos, feature_importance[sorted_idx], align='center')
	featax.set_yticks(pos)
	featax.set_yticklabels(np.array(X_raw.columns)[sorted_idx], fontsize=8)
	featax.set_xlabel('Relative Feature Importance')
	plt.tight_layout()
	plt.show()


def bestFeatures():

	# selectKBest
	# segment is 16th feature by importance
	# apply SelectKBest class to extract top 10 best features
	bestfeatures = SelectKBest(score_func=chi2, k=10)
	fit = bestfeatures.fit(X_raw, y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X_raw.columns)
	# concat two dataframes for better visualization
	featureScores = pd.concat([dfcolumns, dfscores], axis=1)
	featureScores.columns = ['Specs', 'Score']
	print(featureScores.nlargest(16, 'Score'))

	# tree based classifier selection
	from sklearn.ensemble import ExtraTreesClassifier
	model = ExtraTreesClassifier()
	model.fit(X_raw, y)
	print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
	# plot graph of feature importances for better visualization
	feat_importances = pd.Series(model.feature_importances_, index=X_raw.columns)
	feat_importances.nlargest(16).plot(kind='barh')
	plt.show()

	# heatmap for correlations
	# get correlations of each features in dataset
	corrmat = data.corr()
	top_corr_features = corrmat.index
	plt.figure(figsize=(25, 25))
	# plot heat map
	print(data[top_corr_features].corr()[["Segment", "GameWon"]])
	g = sns.heatmap(data[top_corr_features].corr(), fmt=".2f", annot=True, cmap="RdYlGn")
	plt.show()


def main():
	# if yore going to run svc, please for the love of god,
	# change the training set size to <20% just so it would finish it this decade
	support_vector_classification()
	# naive_bayes_classification()
	# logistic_regression_classification()
	# bestFeatures()


if __name__ == '__main__':
	main()
