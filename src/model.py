from sklearn.neighbors import NearestNeighbors
import time
import numpy as np
class KNNRecommender:

    def __init__(self,args,matrix_train,matrix_test,user_age_dic,user_gender_dic,model) -> None:
        self.args = args
        self.matrix_train = matrix_train
        self.matrix_test = matrix_test
        #self.user_id=args.user_id
        self.user_age_dic=user_age_dic
        self.user_gender_dic=user_gender_dic

        self.matrix_trainnp=np.array(self.matrix_train)

        self.model=model
        pass

    def fit(self,matrix):
        #fits KNN with matrix_train
        model = NearestNeighbors(n_neighbors=1000,algorithm='auto',metric='cosine') # try  chaiinging metric or algorithm
        model.fit(np.array(matrix))
        return model

    def predict(self,matrix):
        # show row wehre row index is equal to user_id
        # return distances, indices

        model=self.model
        k=min(self.args.k,len(matrix))
        distances, indices = model.kneighbors(self.vector_test.values.reshape(1,-1),n_neighbors=1000)
        similaruids=(matrix)[indices[0]]

        return distances,indices[0], similaruids# this return directlry returns user_id not index (this is done because we want to use user_id as index)
    
    # def get_gender_matrix(self ):
    #     #select same gender as user_id and create user-item matrix
    #     cur_gender=self.user_gender_dic[self.user_id]
    #     users=np.array([i for i in range(len(self.user_gender_dic.keys())) if list(self.user_gender_dic.values())[i]==cur_gender])
    #     #return matrixssss
    #     #gender_matrix=self.matrix_train.loc[users]
    #     gender_matrix=self.matrix_trainnp[users]
        
    #     return gender_matrix
    
    # def get_age_matrix(self ):
    #     # check age matrix getting time

    #     #select similar age as user_id and create user-item matrix
    #     cur_age=self.user_age_dic[self.user_id]
    #     users=[i for i in range(len(self.user_age_dic.keys())) if abs(list(self.user_age_dic.values())[i]-cur_age)<5]
    #     #return matrixssss
    #     age_matrix=self.matrix_trainnp[users]
    #     #print("similar age users: ",users)

    #     return age_matrix

    def getmetric(self):
        #return metric
        #check if recommended products are in test set for user_id
        #calculate precision and recall, f1 score

        actual=list(self.vector[self.vector!=0].keys())


        precision=(len(set(self.recommended_products).intersection(set(actual)))/len(self.recommended_products))
        
        if len(actual)==0:
            recall=0
        else:
            recall=(len(set(self.recommended_products).intersection(set(actual)))/len(actual))
        # if there is recommended product in top 3 of the recommended list, then it is considered as correct recommendation
        top3=self.recommended_products[:2]
        if len(set(top3).intersection(set(actual)))>0:
            accuracy=1
        else:
            accuracy=0

        return precision, recall,accuracy

    def recommend(self,user_id):
        
        self.user_id=user_id
        self.vector=self.matrix_train.loc[self.user_id]
        self.vector_test=self.matrix_test.loc[self.user_id]

        all_distances, all_indices,all_uids = self.predict(self.matrix_trainnp)

        

        topreco = self.matrix_trainnp[all_indices[:]]
        all_top_recommendation = np.sum(topreco[:self.args.topk], axis=0)

        cur_gender = self.user_gender_dic[self.user_id]
        cur_age = self.user_age_dic[self.user_id]

        # Filter users by gender and age without loops
        gender_mask = np.array(list(self.user_gender_dic.values())) == cur_gender
        age_diff = np.abs(np.array(list(self.user_age_dic.values())) - cur_age) < 5

        # Apply the gender and age masks to get relevant users
        gender_top_recommendation = np.sum(topreco[gender_mask][:self.args.topk], axis=0)
        age_top_recommendation = np.sum(topreco[age_diff][:self.args.topk], axis=0)

        # Calculate the final recommendation scores
        avgrecommendation = (
            all_top_recommendation * self.args.org_weight +
            age_top_recommendation * self.args.age_weight +
            gender_top_recommendation * self.args.gender_weight
        )

        product_idx = np.array(self.matrix_train.columns)

        # Sort and get indices of recommended products
        argsort = np.argsort(avgrecommendation)[::-1]
        topproduct = product_idx[argsort]
        topscore = avgrecommendation[argsort]
        #print("Top {} recommendation for user_id: {}".format(self.args.topk,self.user_id))


        self.recommended_products=topproduct[:self.args.topk]
        self.recommended_score=topscore[:self.args.topk]

        
        precision, recall,accuracy=self.getmetric()
        #print("recommendation time: ",time.time()-rec_time)
        #precision, recall,accuracy=0,0,0    


        return self.recommended_products,self.recommended_score,precision, recall,accuracy
