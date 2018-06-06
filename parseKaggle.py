import pandas as pd
import numpy as np


def parse_kaggle():
    data = pd.read_csv("data/kaggle.csv")
    # data = data.dropna()
    data = data[pd.notnull(data['Rating'])]
    data = data[pd.notnull(data['Review Text'])]

    # if  in data["Rating"]:
    #     print(data["Rating"])
    # print(data.describe())

    # print(data["Review Text"])
    # print(data["Rating"])

    # x_train = np.copy(data["Review Text"])
    # y_train = np.divide(data["Rating"], 5)
    #

    # in data["overall"], the rating is given for each corresponding index in data["reviewText"]
    # 1 means negative reponse and 5 means positive response
    neg = data.loc[data["Rating"] == 1]
    pos = data.loc[data["Rating"] == 5][0:neg.shape[0]]

    # parse it into numpy array and store the reviewText in x_train and the rating in y_train
    x_train = np.copy(neg["Review Text"])
    x_train = np.append(x_train, np.copy(pos["Review Text"]))
    y_train = np.subtract(neg["Rating"], 1)
    y_train = np.append(y_train, np.subtract(pos["Rating"], 4))

    print(x_train)
    print(y_train)

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

