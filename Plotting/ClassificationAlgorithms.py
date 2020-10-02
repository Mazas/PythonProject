import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


def main():
	data = pd.read_csv("raw_cluster_labels.csv")
	lb_make = LabelEncoder()
	data["Race_code"] = lb_make.fit_transform(data["Race"])
	races = data.loc[:, ['Race']].values
	data = data.drop("Race", axis=1)

	y = data['GameWon']
	X = data.drop(['GameWon', 'Touchdowns', 'TouchdownsAgainst', 'Score', 'ScoreAgainst'], axis=1)
	print("Entries in the dataset: ", len(data))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

	print("Training set size: ", len(X_train))
	classifier = SVC(kernel='linear')
	print("Fitting training set...")
	classifier.fit(X_train, y_train)

	print("Predicting...")
	y_predict = classifier.predict(X_test)

	print(confusion_matrix(y_test, y_predict))
	print(classification_report(y_test, y_predict))


if __name__ == '__main__':
	main()
