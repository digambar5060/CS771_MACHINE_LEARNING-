import numpy as np

# calculate means of each seen classes
t_data = np.load('X_seen.npy', allow_pickle=True, encoding='bytes')
seen_class_means = [np.mean(class_data, axis=0) for class_data in t_data]
seen_class_means = np.array(seen_class_means)

# calculate similarity of unseen classes with seen classes
seen_class_attributes = np.load('class_attributes_seen.npy')
seen_class_attributes = np.array(seen_class_attributes)
unseen_class_attributes = np.load('class_attributes_unseen.npy')
unseen_class = np.array(unseen_class_attributes)
similarity_scores = np.dot(unseen_class_attributes, seen_class_attributes.T)

# normalize
all_sum = similarity_scores.sum(axis=1, keepdims=True)
normalize_score = similarity_scores / all_sum

# find mean of unseen class
unseen_class_means = []
for unseen_class_idx in range(10):
    similarity_scores_for_class = normalize_score[unseen_class_idx]
    unseen_class_mean = np.dot(similarity_scores_for_class, seen_class_means)
    unseen_class_means.append(unseen_class_mean)
unseen_class_means = np.array(unseen_class_means)


# predict the correct class of test input and also calculate accuracy
def predict(unseen_mean, x_test, y_test, delta):
    dist = np.zeros((y_test.shape[0], unseen_mean.shape[
        0]))  # make a 2d distace array size of test data and unseen mean and at starting all entry are 0
    for i in range(unseen_mean.shape[0]):
        len = np.dot(np.square(unseen_mean[i] - x_test), delta)
        dist[:, i] = len.reshape(len.shape[0], )

    y_pred = np.argmin(dist, axis=1).reshape(np.argmin(dist, axis=1).shape[0], 1) + 1
    acc = 1 - np.count_nonzero(y_pred - y_test) / float(y_test.shape[0])
    return y_pred, acc


# calculate accuracy
delta = np.ones((seen_class_means.shape[1], 1))
delta = delta / np.sum(delta)

x_test = np.load('Xtest.npy')
x_test=np.array(x_test)
y_test = np.load('Ytest.npy')
y_test=np.array(y_test)

def runner():
    y_pred, acc = predict(unseen_class_means, x_test, y_test, delta)
    print("Accuracy for method 1:", (100 * acc))

runner()
