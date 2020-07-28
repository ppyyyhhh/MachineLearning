import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock


class KNN:

    def __init__(self, k=3, is_classification=True, distance_name='Euclidean'):  # __init__ is the constructor function in python.
        # When, later on in your code, you create an object with "variable = KNN(), this function is called"
        self.k = k
        self.distance_name = distance_name
        self.is_classification = is_classification

        # initializing a variable to store the values of the X and y that will be fitted to the model.
        # Remember: KNN does not have to train the model, just store their values. Use these variables to do that
        self.X = None
        self.y = None

    def _compute_distance(self, x1, x2, distance_name='Euclidean'):
        # Here you will implement your distance function ie. Euclidean distance.
        if distance_name == 'Euclidean':
            distance = euclidean(x1, x2)  # implement the formula for Euclidean Distance -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean

        if distance_name == 'Manhattan':  # you can change this for other kinds of distances
            distance = cityblock(x1, x2)  # implement the formula for Manhattan Distance here

        return distance

    def fit(self, X, y):
        # Remember, you do not have to train a model for KNN, but mainly store the variables
        self.X = X
        self.y = y

    def predict(self, x):
        # For this exercise, we will do the prediction for a single instance.
        # x here is just a single instance.
        distances = []
        # Complete this part of the code.

        # Step 1: You will compare the distance of x to each other instance in your training data.
        # Use the function implemented in _compute_distances
        for training_x in self.X:
            distances.append(self._compute_distance(x, training_x))

        # Step 2: Now that you have the distances, get the labels of each one of the k-nearest instances
        # Hint: the class numpy.argsort returns the index of the sorted values, it might be useful here
        ordered_indexes = np.argsort(distances)  # sorting

        label_of_neighbors = []
        for idx in ordered_indexes[:self.k]:  # from index 0 to index k-1
            label_of_neighbors.append(self.y[idx])  # passing indexes idx to y
        print(label_of_neighbors)

        # Step 3: Make the prediction
        if self.is_classification:
            return Counter(label_of_neighbors).most_common(1)[0][0]  # For classification, you will return the most common class among the nearest neighbors
        else:
            return  sum(label_of_neighbors)/len(label_of_neighbors) # For regression, you will return the mean of the values of the nearest neighbors