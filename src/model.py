from sklearn.neighbors import NearestNeighbors

class KNNRecommender:

    def __init__(self,args,matrix_train,matrix_test) -> None:
        self.args = args
        self.matrix_train = matrix_train
        self.matrix_test = matrix_test
        pass

    def fit(self,matrix):
        #fits KNN with matrix_train
        model = NearestNeighbors(n_neighbors=self.args.k,algorithm='brute',metric='cosine') # try  chaiinging metric or algorithm
        model.fit(matrix)
        return model

    def predict(self,matrix,user_id):
        # show row wehre row index is equal to user_id
        # return distances, indices
        vector=self.matrix_train[self.matrix_train.index==user_id]
        model=self.fit(matrix)
        distances, indices = model.kneighbors(vector.values.reshape(1,-1),n_neighbors=self.args.k)
        similaruids=self.matrix_test.index[indices[0]]
        return distances,indices[0], similaruids# this return directlry returns user_id not index (this is done because we want to use user_id as index)
    
    def same_gender_users(self,user_id, user_gender_dic ):
        #select same gender as user_id and create user-item matrix
        cur_gender=user_gender_dic[user_id]
        users=[list(user_gender_dic.keys())[i] for i in range(len(user_gender_dic.keys())) if list(user_gender_dic.values())[i]==cur_gender]
        #return matrixssss
        self.gender_matrix=self.matrix_train.loc[users]
        
        return users
    
    def similar_age_users(self,user_id, user_age_dic ):
        #select similar age as user_id and create user-item matrix
        cur_age=user_age_dic[user_id]
        users=[list(user_age_dic.keys())[i] for i in range(len(user_age_dic.keys())) if abs(list(user_age_dic.values())[i]-cur_age)<5]
        #return matrixssss
        self.age_matrix=self.matrix_train.loc[users]
        print("similar age users: ",users)
        return users

    def recommend(self,user_id):
        distances, indices,uids = self.predict(user_id)
        #top10matrix=self.matrix_train[self.matrix_train.index==indices]
        toprecommendation=self.matrix_train.iloc[indices[:]].sum(axis=0).sort_values(ascending=False).keys()[:]

        print("Top 10 recommendation for user_id: ",user_id)
        print("top similar user ids: ",uids)
        print(toprecommendation)

        pass