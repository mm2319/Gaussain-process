import numpy as np
def obtain_train_data_NonLinear( result_1, result_2, num_samples, Y ):
    x_1_train = []
    num_samples=num_samples
    y_1_train = []
    x_2_train = []
    y_2_train = []

    for i in range(num_samples):
            u = Y[i,0]
            v = Y[i,1]

            x = np.array([1, u, v, u**2, v**2, u*v, u**3, v**3])
            x_1_train.append(x)
            y_1_train.append(result_1[i])
            x_2_train.append(x)
            y_2_train.append(result_2[i])


    return x_1_train, y_1_train, x_2_train, y_2_train