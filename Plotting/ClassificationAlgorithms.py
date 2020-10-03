import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("raw_with_cluster_labels.csv")
lb_make = LabelEncoder()
data["Race_code"] = lb_make.fit_transform(data["Race"])
races = data.loc[:, ['Race']].values
data = data.drop("Race", axis=1)

y = data['GameWon']
X = data.drop(['GameWon', 'Touchdowns', 'TouchdownsAgainst', 'Score', 'ScoreAgainst'], axis=1)
print("Entries in the dataset: ", len(data))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)


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


def main():
	# support_vector_classification()
	naive_bayes_classification()


if __name__ == '__main__':
	main()
