import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def results_df(labels, pred, probas, score):
	print(f"Mean score: {score}")
	df = pd.DataFrame({
		'label' : labels,
		'pred' : pred, 
		'class 1 prob' : probas[0],
		'class 2 prob' : probas[1],
		'class 2 prob' : probas[2],
	})
	print(df)
	return df

def one_point_two():
	X, y = load_iris(return_X_y=True)

	clf = LogisticRegression(random_state=0, max_iter=1000)
	clf = clf.fit(X, y)

	pred = clf.predict(X)
	probas = clf.predict_proba(X).transpose()
	score = clf.score(X, y)

	results_df(y, pred, probas, score)

def one_point_three():
	X, y = load_iris(return_X_y=True)

	clf = GaussianNB()
	clf.fit(X, y)

	pred = clf.predict(X)
	probas = clf.predict_proba(X).transpose()
	score = clf.score(X, y)

	results_df(y, pred, probas, score)

if __name__ == "__main__":
	#one_point_two()
	one_point_three()