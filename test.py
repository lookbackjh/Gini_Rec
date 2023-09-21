import argparse
from src.datagenerator import DataGenerator
from src.model import KNNRecommender
import pandas as pd

def parser():

    args = argparse.ArgumentParser()
    args.add_argument('--train_ratio', type=float, default=0.8)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--k', type=int, default=5)
    args.add_argument('--topk', type=int, default=5)
    args.add_argument('--cur_year', type=int, default=2023)
    args = args.parse_args()
    return args

def recommender(args,df):
    datagenerator = DataGenerator(args,df)
    matrix_train,matrix_test=datagenerator.generate_user_item_matrix()
    gendermap,agemap=datagenerator.user_feature_dictionary()

    rec = KNNRecommender(args,matrix_train,matrix_test)
    rec.fit()
    rec.recommend(7) # user_id = 7
    rec.similar_age_users(7,agemap)



    pass



if __name__ == "__main__":
    args = parser()
    df = pd.read_pickle("order_info_frequency_1000.pickle")
    recommender(args,df)
