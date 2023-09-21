import argparse
from src.datagenerator import DataGenerator
from src.model import KNNRecommender
import pandas as pd
import tqdm
import numpy as np

def parser():

    args = argparse.ArgumentParser()
    args.add_argument('--train_ratio', type=float, default=0.8)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--k', type=int, default=10)
    args.add_argument('--topk', type=int, default=10)
    args.add_argument('--cur_year', type=int, default=2023)
    args.add_argument('--user_id', type=int, default=7)
    args = args.parse_args()
    return args

def recommender(args,df):
    datagenerator = DataGenerator(args,df)
    matrix_train,matrix_test=datagenerator.generate_user_item_matrix()
    gendermap,agemap=datagenerator.user_feature_dictionary()

    precisions=[]
    recalls=[]
    f1scores=[]

    for userid in tqdm.tqdm(list(gendermap.keys())[:500]):
        args.user_id=userid 
        rec = KNNRecommender(args,matrix_train,matrix_test,agemap,gendermap)
        precsion,recall, f1score=rec.recommend() # user_id = 7

    print("precision: ",np.mean(np.array(precisions)))
    print("recall: ",np.mean(np.array(recalls)))
    print("f1score: ",np.mean(np.array(f1scores)))

    pass



if __name__ == "__main__":
    args = parser()
    df = pd.read_pickle("order_info_frequency_1000.pickle")
    recommender(args,df)
