from sklearn.neighbors import KNeighborsClassifier


class CreditModel:
    def __init__(self, n=9):
        """
        Instantiates the model object, creating class variables if needed.
        """

        # TODO: Initialize your model object.
        self.knn = KNeighborsClassifier(n_neighbors=n, weights='uniform', algorithm='brute')

    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """

        # TODO: Fit your model based on the given X and y.
        self.knn.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """

        # TODO: Predict on `X_test` based on what you learned in the fit phase.
        predictions = self.knn.predict(X_test)
        return predictions
