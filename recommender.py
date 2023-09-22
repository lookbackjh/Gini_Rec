import argparse
from src.datagenerator import DataGenerator
from src.model import KNNRecommender
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import tqdm
import numpy as np
import json
import time
def parser():

    args = argparse.ArgumentParser()
    args.add_argument('--train_ratio', type=float, default=0.7)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--k', type=int, default=10)
    args.add_argument('--topk', type=int, default=5)
    args.add_argument('--cur_year', type=int, default=2023)
    args.add_argument('--user_id', type=int, default=7)
    args.add_argument('--age_weight', type=float, default=0.3)
    args.add_argument('--gender_weight', type=float, default=0.3)
    args.add_argument('--org_weight', type=float, default=0.3)
                
    args = args.parse_args()
    return args

def recommender(args,df):
    datagenerator = DataGenerator(args,df)
    matrix_train,matrix_test=datagenerator.generate_user_item_matrix()
    gendermap,agemap=datagenerator.user_feature_dictionary()


    if args.user_id not in list(matrix_train.index):
        raise ValueError("user_id not in train set")


    model = NearestNeighbors(n_neighbors=1000,algorithm='auto',metric='cosine')
    model.fit(np.array(matrix_train))
    rec = KNNRecommender(args,matrix_train,matrix_test,agemap,gendermap,model)
    



# want to save results as json file
    with open('result.txt', 'w') as f:

        for i in tqdm.tqdm(list(gendermap.keys())):        
            product,score,precsion,recall,accuracy=rec.recommend(i) # user_id = 7


            print("accuracy: ",np.mean(np.array(accuracy)))
            print("precision: ",np.mean(np.array(precsion)))
            print("recall: ",np.mean(np.array(recall)))
            print("recommended products: ",product)
            print("recommended score: ",score)

            # f.write every results same line
            f.write("user_id:{} accuracy:{} precision:{} recall:{} recommended products:{} recommended score:{}\n".format(i,np.mean(np.array(accuracy)),np.mean(np.array(precsion)),np.mean(np.array(recall)),product,score))


    #print("recall: ",np.mean(np.array(recalls)))
    #print("f1score: ",np.mean(np.array(f1scores)))

    pass
    # args.user_id=7 #predefined user_id
    # rec = KNNRecommender(args,matrix_train,matrix_test,agemap,gendermap)
    # product,score,precsion,recall, f1score=rec.recommend() # user_id = 7
    # print("recommended products: ",product)
    # print("recommended score: ",score)
    # print("precision: ",precsion)
    # print("recall: ",recall)
    # print("f1score: ",f1score)
    #return np.mean(np.array(precisions)),np.mean(np.array(recalls)),np.mean(np.array(accuracies)), product,score



if __name__ == "__main__":
    args = parser()
    df = pd.read_pickle("order_info_frequency_1000.pickle")
    # for each combination, run recommender and save results
    #change user_id based on your preference

    recommender(args,df)




    # save results



        
