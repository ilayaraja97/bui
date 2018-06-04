import numpy as np

def splitData(data, split_value=0.5):
    """"
    splitData(data, split_value)

        split the data into a training set and a validation set
        split_value is fraction of samples used for validation set
    """

    indices = np.arange(data[0].shape[0])
    np.random.shuffle(indices)

    X = data[0][indices]
    y = data[1][indices]
    nb_validation_samples = int(split_value * X.shape[0])

    x_train = X[:-nb_validation_samples]
    y_train = y[:-nb_validation_samples]
    x_val = X[-nb_validation_samples:]
    y_val = y[-nb_validation_samples:]

    return x_train,y_train,x_val,y_val