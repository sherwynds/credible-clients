# credible-clients

This repository contains a small pre-task for potential ML team
members for [UBC Launch Pad](https://www.ubclaunchpad.com).


## Overview

The dataset bundled in this repository contains information about credit card
bill payments, courtesy of the [UCI Machine Learning Repository][UCI]. Your task
is to train a model on this data to predict whether or not a customer will default
on their next bill payment.

Most of the work should be done in [**`model.py`**](model.py). It contains a
barebones model class; your job is to implement the `fit` and `predict` methods,
in whatever way you want (feel free to import any libraries you wish). You can
look at [**`main.py`**](main.py) to see how these methods will be called. Don't
worry about getting "good" results (this dataset is _very tough_ to predict on)
— treat this as an exploratory task!

To run this code, you'll need Python and three libraries: [NumPy], [SciPy],
and [`scikit-learn`]. After invoking **`python main.py`** from your shell of
choice, you should see the model accuracy printed: approximately 50% if you
haven't changed anything, since the provided model predicts completely randomly.

## Instructions

Here are the things you should do:

1. Fork this repo, so we can see your code!
2. Install the required libraries using `pip install -r requirements.txt` (if needed).
3. Ensure you see the model's accuracy/precision/recall scores printed when running `python main.py`.
4. Replace the placeholder code in [`model.py`](model.py) with your own model.
5. Fill in the "write-up" section below in your forked copy of the README.

_Good luck, and have fun with this_! :rocket:


## Write-up

I chose to do a KNN approach since I am just starting out with machine learning but am passionate to learn more. My value for K was 9, since that seemed to produce the best accuracy, precision and recall values. I could increase the accuracy at higher values of K, or using SVM Classification methodology, but I found this reduced the recall to an insignificant amount, hence I opted for the first approach in my final submission. I tried to play around with it using tensorflow, but have not had time to come to a complete solution. 


## Data Format

`X_train` and `X_test` contain data of the following form:

| Column(s) | Data |
| :-------: | ---- |
| 0         | Amount of credit given, in dollars |
| 1         | Gender (_1 = male, 2 = female_) |
| 2         | Education (_1 = graduate school; 2 = university; 3 = high school; 4 = others_) |
| 3         | Marital status (_1 = married; 2 = single; 3 = others_) |
| 4         | Age, in years |
| 5–10      | History of past payments over 6 months (_-1 = on-time; 1 = one month late; …_) |
| 11–16     | Amount of previous bill over 6 months, in dollars |
| 17–22     | Amount of previous payment over 6 months, in dollars |

`y_train` and `y_test` contain a `1` if the customer defaulted on their next
payment, and a `0` otherwise.


[UCI]: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
[NumPy]: http://www.numpy.org
[SciPy]: https://www.scipy.org
[`scikit-learn`]: http://scikit-learn.org
