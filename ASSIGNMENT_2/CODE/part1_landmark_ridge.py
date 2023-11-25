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


hp = [2, 5, 20, 50, 100]

for ii in hp:
    z = np.random.choice(x_train_data, ii, replace=False)
    Id = np.identity(ii)
    x_reshaped = x_train_data[:, np.newaxis]
    y_reshaped = z[np.newaxis, :]
    squared_difference = (x_reshaped - y_reshaped) ** 2
    xf_train_data = np.exp(-0.1 * squared_difference)

    XT_X = np.dot(xf_train_data.T, xf_train_data)
    lambda_identity = 0.1 * np.identity(XT_X.shape[0])
    XT_y = np.dot(xf_train_data.T, y_train_data.reshape((-1, 1)))
    W = np.dot(np.linalg.inv(XT_X + lambda_identity), XT_y)

    x_reshaped = x_test_data[:, np.newaxis]
    y_reshaped = z[np.newaxis, :]
    squared_difference = (x_reshaped - y_reshaped) ** 2
    xf_test_data = np.exp(-0.1 * squared_difference)
    y_pred = np.dot(xf_test_data, W)

    errors = y_test_data[:, np.newaxis] - y_pred
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    print(f'RMSE score  for lambda = {ii} is {rmse}')

    plt.figure(ii)
    plt.title('For lambda = ' + str(ii) + ', rmse = ' + str(rmse))
    plt.plot(x_test_data, y_pred, 'r*', label='Predicted')
    plt.plot(x_test_data, y_test_data, 'b*', label='Real')
    plt.legend()

plt.show()