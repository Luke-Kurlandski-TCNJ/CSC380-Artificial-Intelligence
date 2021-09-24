from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

def one_point_four():
	X, y = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.25, random_state=0)

	clf = LinearSVC(max_iter=10000)

	print(mean(cross_val_score(clf, X, y, cv=5)))

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
	plt.savefig("one_point_four.png")

if __name__ == "__main__":
	#one_point_two()
	one_point_three()
	one_point_four()