# How to Use

## 1. Data Generator

As there are too many data in the dataset we only select some of the top 1000 user 
and use their data to train. 
to see the details of how we have made the `order_info_frequency_1000.pickle` data, check `datagenerator.ipynb`

## 2. Model Train and Recommendation

`recommender.py` utilizes a KNN-based algorithm trained on our model to provide product recommendations. Adjust the `topk` parameter to specify the number of recommended products you wish to receive based on your preferences.

