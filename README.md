# DecisionTree
Implementation decision tree algorithm with C++11 and apply it to classification task on UCI-IRIS dataset.

The strategy used to select attribute is based on information gain -- ID3 decision tree[Quinlan, 1986]. I'll or you can change/update the strategy in function *chooseAttr()*.

# Dataset
Iris Data Set[http://archive.ics.uci.edu/ml/datasets/Iris]. Or you can download dataset with 'downloadDataset.ipynb' and save to csv file.

For continuous value is impossible to divide nodes directly, this project adopted a simple and crude method(*np.trunc()*) to do discretization operation. A better way is dichotomy strategy just as described in the book "Machine Learning".

# Contribution
This is the first project which implemente decision tree with C++ and apply it to classification task on UCI-IRIS dataset.

# License
MIT License

Copyright (c) 2021 TongJiayan

