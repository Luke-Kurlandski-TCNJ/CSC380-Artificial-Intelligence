from pathlib import Path
import re
from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
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

def clean_data(path):
	new_path = Path(path.as_posix().replace(path.stem, path.stem + "_clean"))
	new_lines = []
	with open(path, 'r') as f_r:
		for l in f_r:
			new_l = l.lstrip()
			new_l = new_l.replace("  ", ",")
			new_l = new_l.replace(" ", ",")
			new_lines.append(new_l)

	with open(new_path, 'w') as f_w:
		f_w.writelines(new_lines)

def plot_iris(save_file, X, y):

	features = ["SL", "SW", "PL", "PW"]

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
			axs[i, j].scatter(X[:,i], X[:,j], c=colors, marker=".")
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
	plot_iris("HW2/iris_bayes_pred.png", X_test, pred)
	plot_iris("HW2/iris_bayes_truth.png", X_test, y_test)

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

	pred = clf.predict(X_test)
	probas = clf.decision_function(X_test).transpose()
	print(results_df(y_test, pred, probas))

	# Plot the iris predictions.
	plot_iris("HW2/iris_svm_pred.png", X_test, pred)
	plot_iris("HW2/iris_svm_truth.png", X_test, y_test)
		
def one_point_five():
    # Load the digits data, partition into train and test splits.
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # The linear SVM classifier.
    clf = LinearSVC(max_iter=100000)

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

    print(f"Digits | Average CV accuracy: {mean(scores)}")

def one_point_six():
	path_to_dataset = Path("/home/hpc/kurlanl1/CSC-380/CSC380-Artificial-Intelligence/UCIHARDataset/")

	clean_data(path_to_dataset / "train/X_train.txt")
	clean_data(path_to_dataset / "test/X_test.txt")

	X_train = np.genfromtxt(path_to_dataset / "train/X_train_clean.txt", delimiter=',')
	y_train = np.genfromtxt(path_to_dataset / "train/y_train.txt", delimiter=',')
	X_test = np.genfromtxt(path_to_dataset / "test/X_test_clean.txt", delimiter=',')
	y_test = np.genfromtxt(path_to_dataset / "test/y_test.txt", delimiter=',')

	# The linear SVM classifier.
	clf = LinearSVC(max_iter=100000)

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

	print(f"Digits | Average CV accuracy: {mean(scores)}")

	print(f"Net model accuracy: {clf.score(X_test, y_test)}")

if __name__ == "__main__":
	#one_point_two()
	#one_point_three()
	#one_point_four()
	#one_point_five()
	one_point_six()
