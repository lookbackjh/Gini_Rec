import argparse
from src.datagenerator import DataGenerator
import pandas as pd

def parser():

    args = argparse.ArgumentParser()
    args.add_argument('--train_ratio', type=float, default=0.8)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--k', type=int, default=5)
    args.add_argument('--topk', type=int, default=5)
    args = args.parse_args()
    return args

if __name__ == "__main__":
    args = parser()
    df = pd.read_pickle("order_info_frequency.pickle")
    datagenerator = DataGenerator(args,df)
    datagenerator.generate_user_item_matrix()
