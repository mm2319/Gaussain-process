import numpy as np
def obtain_train_data_Two_compart( result_1, result_2, num_samples, Y):
    num_samples=num_samples
    x_1_train = []
    y_1_train = []
    x_2_train = []
    y_2_train = []

    for i in range(num_samples):
            N = Y[i,0]
            K = Y[i,1]

            x = np.array([1., N, K,N*K, N**2, K**2,(N**2)/(K), K*(N**(2/3))])
            x_1_train.append(x)
            y_1_train.append(result_1[i])
            x_2_train.append(x)
            y_2_train.append(result_2[i])


    return x_1_train, y_1_train, x_2_train, y_2_train