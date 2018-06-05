import pandas as pd
import numpy as np


def parse_amazon():
    data = pd.read_json("data/amazon.json", lines=True)
    data = data[:][:10000]
    # print(data.describe())

    # print(data["reviewText"])
    # print(data["overall"])

    x_train = np.copy(data["reviewText"])
    # 3 -> 0, 4 -> 1
    y_train = np.round(np.divide(data["overall"], 7))

    df = pd.DataFrame(y_train)
    # print(df)
    print(np.divide(df["overall"].value_counts(), 1000.00))  # 2786.77))
    return x_train, y_train


def main():
    parse_amazon()


if __name__ == "__main__":
    main()
