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
    # print(x_train)
    # print(y_train)
    # df = pd.DataFrame(y_train)
    # print("h")
    # print(np.divide(df["overall"].value_counts(), 1))  # 2786.77))
    return x_train, y_train


def main():
    parse_amazon()


if __name__ == "__main__":
    main()
