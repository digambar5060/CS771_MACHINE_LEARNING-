

# Classification Methods README

This README provides an overview of two classification methods for classifying unseen classes using seen class information. These methods aim to categorize data into unseen classes based on their similarity to seen classes. Both methods involve a training phase to compute class means and a testing phase to predict labels for unseen class inputs.

## Method 1

**Approach:**

1. Compute the mean of each seen class using training data.

2. Compute the similarity between each unseen class and all seen classes using a dot product-based method.

3. Normalize the similarity vector of each unseen class to ensure it sums to 1, creating a convex combination.

4. Compute the mean of each unseen class by taking a convex combination of the means of seen classes.

5. Apply the trained model to predict labels for unseen class test inputs.

6. Calculate classification accuracies for evaluation.

**To Run Method 1:**

Execute the `convex.py` file to run Method 1.

## Method 2

**Approach:**

1. Compute the mean of each seen class using training data.

2. Learn a multi-output regression model where the class attribute vector is the input, and the class mean vector is the output. This model utilizes the seen class attributes and their mean vectors for training.

3. Apply the trained regression model to compute the mean of each unseen class.

4. Use the learned regression model to predict labels for unseen class test inputs.

5. Calculate classification accuracies for evaluation.

**Note:** To optimize the regression model, experiment with different values of the regularization hyperparameter (\lambda) during training.

**To Run Method 2:**

Execute the `regress.py` file to run Method 1.
