# How to Use

## 1. Data Generator

As there are too many data in the dataset we only  top 1000 users who have certain amount of purchase record.
and use their data to train. 
to see the details of how we have made the `order_info_frequency_1000.pickle` data, check `datagenerator.ipynb`

## 2. Model Train and Recommendation

`recommender.py` utilizes a KNN-based algorithm trained on our model to provide product recommendations. Adjust the `topk` parameter to specify the number of recommended products you wish to receive based on your preferences.
In our code, we implemented to get accuracy, precision, recommended products and recommend score(which indicates how mush we recommend certain product) for every user. 

