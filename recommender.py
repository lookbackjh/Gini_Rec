import argparse
from src.datagenerator import DataGenerator
from src.model import KNNRecommender
import pandas as pd
import tqdm
import numpy as np
import json

def parser():

    args = argparse.ArgumentParser()
    args.add_argument('--train_ratio', type=float, default=0.9)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--k', type=int, default=20)
    args.add_argument('--topk', type=int, default=10)
    args.add_argument('--cur_year', type=int, default=2023)
    args.add_argument('--user_id', type=int, default=7)
    args.add_argument('--age_weight', type=float, default=0.4)
    args.add_argument('--gender_weight', type=float, default=0.4)
    args.add_argument('--org_weight', type=float, default=0.2)
                
    args = args.parse_args()
    return args

def recommender(args,df):
    datagenerator = DataGenerator(args,df)
    matrix_train,matrix_test=datagenerator.generate_user_item_matrix()
    gendermap,agemap=datagenerator.user_feature_dictionary()




    precisions=[]
    recalls=[]
    accuracies=[]

    for userid in tqdm.tqdm(list(gendermap.keys())[:1000]):
        args.user_id=userid 
        rec = KNNRecommender(args,matrix_train,matrix_test,agemap,gendermap)
        product,score,precsion,recall,accuracy=rec.recommend() # user_id = 7
        precisions.append(precsion)
        recalls.append(recall)
        accuracies.append(accuracy)
        #f1scores.append(f1score)

    print("precision: ",np.mean(np.array(precisions)))
    print("recall: ",np.mean(np.array(recalls)))
    print("accuracy: ",np.mean(np.array(accuracies)))
    #print("f1score: ",np.mean(np.array(f1scores)))


    # args.user_id=7 #predefined user_id
    # rec = KNNRecommender(args,matrix_train,matrix_test,agemap,gendermap)
    # product,score,precsion,recall, f1score=rec.recommend() # user_id = 7
    # print("recommended products: ",product)
    # print("recommended score: ",score)
    # print("precision: ",precsion)
    # print("recall: ",recall)
    # print("f1score: ",f1score)
    return np.mean(np.array(precisions)),np.mean(np.array(recalls)),np.mean(np.array(accuracies))



if __name__ == "__main__":
    args = parser()
    df = pd.read_pickle("order_info_frequency_1000.pickle")
    kparam=[10,15,20,25,30]
    topkparam=[5,7,10]
    train_ratio=[0.7,0.8,0.9]
    # want every combination of kparam, topkparam, train_ratio

    for k in kparam:
        for topk in topkparam:
            for train_ratio in train_ratio:
                args.k=k
                args.topk=topk
                args.train_ratio=train_ratio
                precision,recall,accuracy=recommender(args,df)
                
                #print("k: {}, topk: {}, train_ratio: {}, precision: {}, recall: {}, accuracy: {}".format(k,topk,train_ratio,precision,recall,accuracy))
                with open("result.txt", "a") as f:
                    f.write("k: {}, topk: {}, train_ratio: {}, precision: {}, recall: {}, accuracy: {}\n".format(k,topk,train_ratio,precision,recall,accuracy))


        
