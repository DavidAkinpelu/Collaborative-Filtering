import pandas as pd
import numpy as np
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans, KNNBaseline
from surprise.model_selection import train_test_split
from surprise import accuracy

ratings = pd.read_csv('ratings.csv', nrows=1000000)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.2)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)

# if you wanted to evaluate on the trainset
train_pred = algo.test(trainset.build_testset())
accuracy.rmse(train_pred)

# run the trained model against the testset
test_pred = algo.test(testset)

# get RMSE
print("User-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

param_grid = {'k': [30, 40],
              'bsl_options': {'method': ['als', 'sgd'],
                              'learning_rate': [.0005, 0.0001],
                              'n_epochs': [20, 40],
                              'reg_u': [15, 25],
                              'reg_i': [10, 20]
                              },
              'sim_options': {'name': ['pearson_baseline', 'cosine'],
                              'user_based': [False, True]}
              }
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
gs.fit(data)

algo = gs.best_estimator['rmse']  # pass the best model to algo
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

result = pd.DataFrame(gs.cv_results)
result.to_csv("output_KNN.csv")
