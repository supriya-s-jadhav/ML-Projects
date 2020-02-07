# Demonstrating XGBoost algorithm with imbalanced Classification dataset

## What is an Imbalanced Classification

The imbalanced classification dataset is a dataset where number of observations for each class label is not balanced. Meaning, the class distribution in the dataset is not equal, and is instead biased or skewed.

Many real world classification problems like fraud detection, spam detection etc have an imbalanced class distribution.

Working with imbalanced classification dataset is a challenging problem as most of the machine learning algorithms were designed under the assumption of an equal number of observations in each class. Hence, this results in models that have poor predictive performance.

## XGBoost Machine learning algorithm

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost algorithms performs well in general, even on imbalanced classification dataset, it offers a way to tune the training algorithm to pay more attention to misclassification of the minority class for datasets with a skewed distribution.

## Test case

We will be using make_classification() scikit-learn function to define a synthetic imbalanced two-class classification dataset. We will generate 10,000 observation points with an approximate 1:100 minority to majority class ratio.


## Further Reading

[XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
[mk_classification()](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)