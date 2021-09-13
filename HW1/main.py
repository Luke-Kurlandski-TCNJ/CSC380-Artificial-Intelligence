from pathlib import Path

import numpy as np
import pandas as pd

def one():

	print("1.1")

	# Tidy data represents each example as a row and each attribute in 
		# the row's columns.
	df = pd.DataFrame({
		'A1' : [1, 1, 0, 1, 1],
		'A2' : [0, 0, 1, 1, 1],
		'A3' : [0, 1, 0, 1, 0],
		'y' : [0, 0, 0, 1, 1]
	})
	print(df)

	# Save to a file.
	path = Path("HW1/data.csv")
	df.to_csv(path)

	# Remove from memory, load from saved file, and observe it is
		# identical to the previous dataframe.
	del df
	df = pd.read_csv(path, index_col=0)
	print(df)

def two():
	
	print("1.2")

	# The X, y data format is consistent with sklearn and other popular
		# ML and data science libraries. In this format, each example is
		# a row and columns represent features/attributes. The y array
		# corresponds to each row of X.
	X = np.array([
		[1, 1, 0],
		[1, 0, 1],
		[0, 1, 0],
		[1, 1, 1],
		[1, 1, 0]
	])
	y = np.array([0, 0, 0, 1, 1])

	print(X)
	print(y)

	# We could also use scipy.sparse.csr_matrix if the data contained
		# mostly 0s, or any other sparse matrix format.

def main():
	two()

if __name__ == "__main__":
	main()

