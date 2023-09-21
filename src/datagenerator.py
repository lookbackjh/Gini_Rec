#creates user item matrix and  user-Gender matrix and user-Age matrix
import pandas as pd
class DataGenerator():

    def __init__(self,args,df) -> None:
        self.args = args
        


        self.df=df
        self.userids=df['AUTH_CUSTOMER_ID'].unique()
        self.productids=df['PRODUCT_CODE'].unique()
        

        self.matrix_train=pd.DataFrame(index=self.userids, columns=self.productids)
        self.matrix_train=self.matrix_train.fillna(0)
        self.matrix_test=pd.DataFrame(index=self.userids, columns=self.productids)
        self.matrix_test=self.matrix_test.fillna(0)

        pass
    

    def generate_user_item_matrix(self):
        # df must be already filtered( #상위 몇명만 추출)
        # raise error if df does not have columns: AUTH_CUSTOMER_ID, PRODUCT_CODE
        train_df, test_df = self.train_test_split()
        
        # user-item matrix for train and test
        for index, row in train_df.iterrows():
            self.matrix_train.at[row['AUTH_CUSTOMER_ID'], row['PRODUCT_CODE']]+=1

        for index, row in test_df.iterrows():
            self.matrix_test.at[row['AUTH_CUSTOMER_ID'], row['PRODUCT_CODE']]+=1
        
        return self.matrix_train, self.matrix_test


    def user_feature_dictionary(self):
        self.gendermap={}
        self.agemap={}  
        for user in self.userids:
            temp=self.df[self.df['AUTH_CUSTOMER_ID']==user].iloc[0]
            self.gendermap[user]=temp['GENDER']
            self.agemap[user]=self.args.cur_year-temp['BIRTH_YEAR']
        
        return self.gendermap, self.agemap


    def train_test_split(self):

        train_df = self.df.sample(frac=self.args.train_ratio, random_state=self.args.seed)
        test_df = self.df.drop(train_df.index)
        
        return train_df, test_df
    



