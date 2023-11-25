# import library
import numpy as np

# calculate mean of every seen class by using X_seen data
t_data = np.load('X_seen.npy', allow_pickle=True, encoding='bytes')
seen_class_means = [np.mean(class_data, axis=0) for class_data in t_data]
seen_class_means = np.array(seen_class_means)


# calculate unseen class mean using weight vector dot product with unseen class atribute
# W = imnverse((transpose(As).As + λI))transpose(As)Ms
def unseen_class_mean_calculate(seen_mean ,hyperpara):
    W = np.dot(np.linalg.inv(np.dot(seen_ac.T, seen_ac) + hyperpara * (np.eye(seen_ac.shape[1]))), np.dot(seen_ac.T, seen_mean))
    mean = np.dot(unseen_ac, W)
    return mean


# predict the correct class of test input and also calculate accuracy
def predict(unseen_mean, x_test, y_test, theta):
    dist = np.zeros((y_test.shape[0], unseen_mean.shape[0]))  # make a 2d distace array size of test data and unseen mean and at starting all entry are 0
    for i in range(unseen_mean.shape[0]):
        length = np.dot(np.square(unseen_mean[i]-x_test),theta)
        dist[:, i] = length.reshape(length.shape[0], )

    y_pred = np.argmin(dist, axis=1).reshape(np.argmin(dist, axis=1).shape[0], 1)+1
    acc = 1 - np.count_nonzero(y_pred - y_test) / float(y_test.shape[0])
    return y_pred, acc



#take input of unseen class attributes ,seen_class_attributes,x_test,y_test
theta = np.ones((seen_class_means.shape[1], 1))
seen_ac = np.load('class_attributes_seen.npy')
seen_ac = np.array(seen_ac)
unseen_ac = np.load('class_attributes_unseen.npy')
x_test = np.load('Xtest.npy')
x_test=np.array((x_test))
y_test = np.load('Ytest.npy')
y_test=np.array(y_test)


def runner():
    # calculate accuracy for lambada= 0.01
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,0.01)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "0.01" + ") is: " + str(100 * acc))

    # calculate accuracy for lambada= 0.1
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,0.1)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "0.1" + ") is: " + str(100 * acc))

    # calculate accuracy for lambada= 1
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,1)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "1" + ") is: " + str(100 * acc))

    # calculate accuracy for lambada= 10
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,10)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "10" + ") is: " + str(100 * acc))

    # calculate accuracy for lambada= 20
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,20)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "20" + ") is: " + str(100 * acc))

    # calculate accuracy for lambada= 50
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,50)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "50" + ") is: " + str(100 * acc))

    # calculate accuracy for lambada= 100
    unseen_classmean = unseen_class_mean_calculate(seen_class_means,100)
    y_pred, acc = predict(unseen_classmean, x_test, y_test, theta)
    print("accuracy for lambada (" + "100" + ") is: " + str(100 * acc))

runner()
