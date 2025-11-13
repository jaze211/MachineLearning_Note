from sklearn import datasets
digits = datasets.load_digits()
feature = digits.data
target = digits.target
feature[0]
