import pandas as pd
import numpy as np


def parse_amazon():
    data = pd.read_json("data/amazon.json", lines=True)
    # print(data.loc[data["overall"] == 1])
    neg = data.loc[data["overall"] == 1]
    pos = data.loc[data["overall"] == 5][0:neg.shape[0]]

    # print(neg["reviewText"])
    # print(neg["overall"])

    x_train = np.copy(neg["reviewText"])
    x_train = np.append(x_train, np.copy(pos["reviewText"]))
    # 3 -> 0, 4 -> 1
    y_train = np.subtract(neg["overall"], 1)
    y_train = np.append(y_train, np.subtract(pos["overall"], 4))

    # x_train = []
    # y_train = []
    # i = 0
    # count1 = 0
    # count5 = 0
    # for x in data["overall"]:
    #     if x == 5 and count5 < count1:
    #         x_train.append(data["reviewText"][i])
    #         y_train.append(1)
    #         count5 += 1
    #     if x == 1:
    #         x_train.append(data["reviewText"][i])
    #         y_train.append(0)
    #         count1 += 1
    #     i += 1
    # x_train = np.copy(x_train).transpose()
    # y_train = np.copy(y_train).transpose()
    #
    #
    # # x_train = np.copy(data["reviewText"])
    # # # 3 -> 0, 4 -> 1
    # # y_train = np.round(np.divide(data["overall"], 7))
    # # y_train = np.copy(data["overall"])
    # # print(x_train)
    # # print(y_train)
    # df = pd.DataFrame(y_train)
    # # print(df.describe())
    # #
    # print(np.divide(df[0].value_counts(), 1))  # 2786.77))
    return x_train, y_train


def main():
    parse_amazon()


if __name__ == "__main__":
    main()
