from sklearn.neighbors import NearestNeighbors

class KNNRecommender:

    def __init__(self,args,matrix_train,matrix_test,user_age_dic,user_gender_dic) -> None:
        self.args = args
        self.matrix_train = matrix_train
        self.matrix_test = matrix_test
        self.user_id=args.user_id
        self.user_age_dic=user_age_dic
        self.user_gender_dic=user_gender_dic
        pass

    def fit(self,matrix):
        #fits KNN with matrix_train
        model = NearestNeighbors(n_neighbors=self.args.k,algorithm='brute',metric='cosine') # try  chaiinging metric or algorithm
        model.fit(matrix)
        return model

    def predict(self,matrix):
        # show row wehre row index is equal to user_id
        # return distances, indices
        vector=self.matrix_test[self.matrix_train.index==self.user_id]
        model=self.fit(matrix)
        k=min(self.args.k,len(matrix))
        distances, indices = model.kneighbors(vector.values.reshape(1,-1),n_neighbors=k)
        similaruids=self.matrix_test.index[indices[0]]
        return distances,indices[0], similaruids# this return directlry returns user_id not index (this is done because we want to use user_id as index)
    
    def get_gender_matrix(self ):
        #select same gender as user_id and create user-item matrix
        cur_gender=self.user_gender_dic[self.user_id]
        users=[list(self.user_gender_dic.keys())[i] for i in range(len(self.user_gender_dic.keys())) if list(self.user_gender_dic.values())[i]==cur_gender]
        #return matrixssss
        gender_matrix=self.matrix_train.loc[users]
        
        return gender_matrix
    
    def get_age_matrix(self ):
        #select similar age as user_id and create user-item matrix
        cur_age=self.user_age_dic[self.user_id]
        users=[list(self.user_age_dic.keys())[i] for i in range(len(self.user_age_dic.keys())) if abs(list(self.user_age_dic.values())[i]-cur_age)<5]
        #return matrixssss
        age_matrix=self.matrix_train.loc[users]
        #print("similar age users: ",users)
        return age_matrix

    def getmetric(self):
        #return metric
        #check if recommended products are in test set for user_id
        #calculate precision and recall, f1 score

        actual=list(self.matrix_test.loc[self.user_id][self.matrix_test.loc[self.user_id]!=0].keys())
        precision=(len(set(self.recommended_products).intersection(set(actual)))/len(self.recommended_products))
        recall=(len(set(self.recommended_products).intersection(set(actual)))/len(actual))
        f1score=2*(precision*recall)/(precision+recall)

        return precision, recall, f1score

    def recommend(self):
        all_distances, all_indices,all_uids = self.predict(self.matrix_train)
        all_top_recommendation=self.matrix_train.iloc[all_indices[:]].sum(axis=0)

        age_matrix=self.get_age_matrix()
        age_distances, age_indices,age_uids = self.predict(age_matrix)
        age_top_recommendation=age_matrix.iloc[age_indices[:]].sum(axis=0)

        gender_matrix=self.get_gender_matrix()
        gender_distances,gender_indices,gender_uids=self.predict(gender_matrix)
        gender_top_recommendation=gender_matrix.iloc[gender_indices[:]].sum(axis=0)

        avgrecommendation=all_top_recommendation+age_top_recommendation+gender_top_recommendation
        avgrecommendation=avgrecommendation/3
        avgrecommendation=avgrecommendation.sort_values(ascending=False)
        #print("Top {} recommendation for user_id: {}".format(self.args.topk,self.user_id))

        self.recommended_products=avgrecommendation[:self.args.topk].index.values
        self.recommended_score=avgrecommendation[:self.args.topk].values

        precision, recall,f1score=self.getmetric()


        return self.recommended_products,self.recommended_score,precision, recall,f1score
