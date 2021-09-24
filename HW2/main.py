from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

def results_df(labels, pred, probas):
	df = pd.DataFrame({
		'label' : labels,
		'pred' : pred, 
		'class 1 prob' : probas[0],
		'class 2 prob' : probas[1],
		'class 2 prob' : probas[2],
	})
	return df

def plot_iris(save_file, y=None):
	if y is None:
		X, y = load_iris(return_X_y=True, as_frame=True)
	else:
		X, _ = load_iris(return_X_y=True, as_frame=True)

	X = X.rename(columns=
		{"sepal length (cm)" : "SL", "sepal width (cm)" : "SW",
		"petal length (cm)" : "PL",  "petal width (cm)" : "PW"}
	)

	features = X.columns.tolist()
	fig, axs = plt.subplots(4, 4)

	for i in range(len(features)):
		for j in range(len(features)):
			colors = []
			for label in y:
				if label == 0:
					colors.append('red')
				elif label == 1:
					colors.append('blue')
				elif label == 2:
					colors.append('green')
			axs[i, j].scatter(X[X.columns[i]].to_numpy(), 
				X[X.columns[j]].to_numpy(), c=colors, marker=".")
			axs[i, j].set_title(f"{features[j]} vs {features[i]}")

	for ax in axs.flat:
		ax.label_outer()
	
	fig.tight_layout()
	fig.savefig(save_file)

def one_point_two():
	X, y = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.25, random_state=0)

	clf = LogisticRegression(random_state=0, max_iter=1000)
	clf = clf.fit(X_train, y_train)

	print(f"Score: {clf.score(X_test, y_test)}")

	pred = clf.predict(X_test)
	probas = clf.predict_proba(X_test).transpose()
	print(results_df(y_test, pred, probas))

def one_point_three():
	X, y = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.25, random_state=0)

	clf = GaussianNB()
	clf.fit(X_train, y_train)

	print(f"Score: {clf.score(X_test, y_test)}")

	train_sizes, train_scores, valid_scores = \
		learning_curve(clf, X_train, y_train, cv=5)

	train_scores_mean = np.mean(train_scores, axis=1)
	valid_scores_mean = np.mean(valid_scores, axis=1)

	plt.close()
	plt.plot(train_sizes, train_scores_mean, label="Training Set")
	plt.plot(train_sizes, valid_scores_mean, label="CV Set")
	plt.legend()
	plt.grid()
	plt.title("Learning Curve for Iris Dataset")
	plt.xlabel("Training Data")
	plt.ylabel("Accuracy")
	plt.savefig("one_point_three.png")

	pred = clf.predict(X_test)
	probas = clf.predict_proba(X_test).transpose()

	print(results_df(y_test, pred, probas))

	# Plot the iris predictions
	plot_iris("iris_bayes.png", clf.predict(X))

def one_point_four():

	# Load the iris data, partition into train and test splits.
	X, y = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.25, random_state=0)

	# The linear SVM classifier.
	clf = LinearSVC(max_iter=10000)

	# A K-Folds cross-validator that helps with partitioning the data.
	kf = KFold(n_splits=5)

	# Stores the accuracy of the classifier on each cross fold.
	scores = []

	# Perform the cross validation.
	for train_index, test_index in kf.split(X_train):
		X_trainCV, X_testCV = X_train[train_index], X_train[test_index]
		y_trainCV, y_testCV = y_train[train_index], y_train[test_index]
		clf.fit(X_trainCV, y_trainCV)
		score = clf.score(X_testCV, y_testCV)
		scores.append(score)

	print(f"Average CV accuracy: {mean(scores)}")

	# Plot the iris predictions
	plot_iris("iris_svm.png", clf.predict(X))

if __name__ == "__main__":
	plot_iris("iris_default.png")
	one_point_two()
	one_point_three()
	one_point_four()