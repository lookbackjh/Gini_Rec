from sklearn.neighbors import NearestNeighbors

class KNNRecommender:

    def __init__(self,args,matrix_train,matrix_test) -> None:
        self.args = args
        self.matrix_train = matrix_train
        self.matrix_test = matrix_test
        pass

    def fit(self):
        #fits KNN with matrix_train
        self.model = NearestNeighbors(n_neighbors=self.args.k,algorithm='brute',metric='cosine') # try  chaiinging metric or algorithm
        self.model.fit(self.matrix_train)
        pass

    def predict(self,user_id):
        distances, indices = self.model.kneighbors(self.matrix_test[user_id],n_neighbors=self.args.k)
        return distances,indices
    
    def recommend(self,user_id):
        distances, indices = self.predict(user_id)
        print(distances)
        print(indices)
        pass