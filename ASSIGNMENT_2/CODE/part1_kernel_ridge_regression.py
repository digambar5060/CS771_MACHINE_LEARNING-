import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# Load data into a Pandas DataFrame
df_train = pd.read_csv('ridgetrain.txt', delimiter='  ', header=None, names=['Feature', 'Target'])
df_test = pd.read_csv('ridgetest.txt', delimiter='  ', header=None, names=['Feature', 'Target'])
x_train_data = df_train['Feature'].values
y_train_data = df_train['Target'].values
x_test_data = df_test['Feature'].values
y_test_data = df_test['Target'].values

def calculate_reshaped_dot_product(alpha, K_test):
    dot_product = np.dot(alpha.T, K_test)
    reshaped_result = dot_product.reshape((-1, 1))
    return reshaped_result



#make kernel=>
# Calculate squared pairwise distances
# Compute the Gaussian kernel matrix
pairwise_sq_dists = np.square(x_train_data[:, np.newaxis] - x_train_data[np.newaxis, :])
gamma = 0.1
K = np.exp(-gamma * pairwise_sq_dists)

lambda_value = [0, 0.1, 1, 10, 100]
In = np.identity(len(x_train_data))

for ii in lambda_value:
    alpha = np.linalg.inv(K + ii * In) @ y_train_data.reshape((-1, 1))
    pairwise_sq_dists = np.square(x_train_data[:, np.newaxis] - x_test_data[np.newaxis, :])
    gamma = 0.1
    K_test = np.exp(-gamma * pairwise_sq_dists)
    y_pred =calculate_reshaped_dot_product(alpha,K_test)

    #print rmse score
    errors = y_test_data[:, np.newaxis] - y_pred
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    print(f'RMSE score  for lambda = {ii} is {rmse}')

    plt.figure(ii)
    plt.title('For lambda = ' + str(ii) + ', rmse = ' + str(rmse))
    plt.plot(x_test_data, y_pred, 'r*',label='Predicted')
    plt.plot(x_test_data, y_test_data, 'b*',label='Real')
    plt.legend()

plt.show()