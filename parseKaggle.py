import pandas as pd
import numpy as np


def parse_kaggle():
    data = pd.read_csv("data/kaggle.csv")
    data = data.dropna()
    # print(data.describe())

    # print(data["Review Text"])
    # print(data["Rating"])

    x_train = np.copy(data["Review Text"])
    y_train = np.divide(data["Rating"], 5)

    # print(x_train.shape)
    # print(y_train.shape)
    # df = pd.DataFrame(y_train)
    # # print(df)
    # print(np.divide(df["Rating"].value_counts(), 2786.77))
    return x_train, y_train


def main():
    parse_kaggle()


if __name__ == "__main__":
    main()

