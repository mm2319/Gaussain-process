import numpy as np
def obtain_train_data_Lorenz( result_1, result_2, result_3, num_samples,y):
    num_samples=num_samples
    x_1_train = []
    y_1_train = []
    x_2_train = []
    y_2_train = []
    x_3_train = []
    y_3_train = []
    for i in range(num_samples):
            X = y[i,0]
            Y = y[i,1]
            Z = y[i,2]
            x = np.array([1., X, Y, Z, X**2, Y**2, Z**2, X*Y, X*Z, Z*Y, X**3, Y**3, Z**3, X*Y*Z, X**4, Y**4, Z**4])
            x_1_train.append(x)
            y_1_train.append(result_1[i])
            x_2_train.append(x)
            y_2_train.append(result_2[i])
            x_3_train.append(x)
            y_3_train.append(result_3[i])
    return x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train