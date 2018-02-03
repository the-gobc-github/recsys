import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
from tabulate import tabulate
import scipy.sparse as sp
from scipy.sparse.linalg import svds

np.random.seed(42)

def rmse(prediction, ground_truth):

    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    indexes = np.argsort(ground_truth)
    print indexes
    plt.plot(prediction[indexes])
    plt.plot(ground_truth[indexes])
    plt.plot(np.convolve(prediction, np.ones((1000,))/1000, mode='valid'))
    plt.plot()
    plt.show()
    return sqrt(mean_squared_error(prediction, ground_truth))


def predict(ratings, similarity, type_='user'):
    if type_ == 'user':
        mean_user_rating = np.array([ np.mean([r for r in rates if r > 0]) for rates in ratings])
        ratings_diff = np.array([ [r-mean if r>0 else 0 for r in rate] for rate,mean in zip(ratings,mean_user_rating)])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type_ == 'item':
        mean_user_rating = np.array([ np.mean([r for r in rates if r > 0]) for rates in ratings])
        pred = mean_user_rating[:, np.newaxis] + ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def recommend_movies(predictions, userID, movies_df, original_ratings_df, test_matrix, num_recommendations=5, example=False):

    predictions_df = pd.DataFrame(predictions)
    predictions_df.index.name = 'UserID'
    predictions_df.columns.name = 'MovieID'

    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').sort_values(['Rating'], ascending=False))

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations=(movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(),how = 'left',left_on = 'MovieID',
        right_on = 'MovieID').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
                     )
    test_reco = (movies_df[movies_df['MovieID'].isin(user_full['MovieID'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(),how = 'left',left_on = 'MovieID',
        right_on = 'MovieID').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False)
                     )

    test_df = pd.DataFrame(test_matrix)
    test_df.index.name = 'UserID'
    test_df.columns.name = 'MovieID'
    usertest = test_df.ix[user_row_number,:]
    indexes = [l+1 for l in list(usertest[usertest>0].index)]
    true_test = user_full.loc[user_full['MovieID'].isin(indexes)]
    pred_test = test_reco.loc[test_reco['MovieID'].isin(indexes)]

    true_test = reformat(true_test,genres=genres)
    pred_test = reformat(pred_test,genres=genres)
    print 'item we should recommend', tabulate(true_test, headers='keys',tablefmt='psql')
    print 'item we predicted on this list', tabulate(pred_test, headers='keys',tablefmt='psql')

    user_full = reformat(user_full,genres=genres)
    recommendations = reformat(recommendations,genres=genres)
    if example :
        print 'item already rated', tabulate(user_full.head(10), headers='keys',tablefmt='psql')
    print 'item prediction', tabulate(recommendations, headers='keys',tablefmt='psql')

    return user_full, recommendations


def reformat(df,genres=None):

    movie_tags = []
    for i,m in df.iterrows() :
        tags = ''
        for g in genres :
            if m[g] == 1 :
                tags += g + '/'
        movie_tags.append(tags[:-1])

    for g in genres :
        del df[g]
    df.loc[:,'genres'] = movie_tags
    return df


def traintestset(df,n_users,n_items,prop=0.25):

    train_data, test_data = cv.train_test_split(df, test_size=prop)
    train_data_matrix = np.zeros((n_users, n_items))

    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]
    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    return train_data_matrix, test_data_matrix


def cf_memory_based(train_data_matrix,test_data_matrix,metric='cosine'):

    user_similarity = np.ones((train_data_matrix.shape[0],train_data_matrix.shape[0]))-pairwise_distances(train_data_matrix, metric=metric)
    for i in range(len(user_similarity)):
        user_similarity[i][i] = 0
        print user_similarity[i]

        plt.plot(user_similarity[i])
        user_similarity[i]= (user_similarity[i] - np.array([min(user_similarity[i])]))/(np.array([max(user_similarity[i])])-np.array([min(user_similarity[i])]))
        print user_similarity[i]
        plt.plot(user_similarity[i])
        plt.show()
        exit()
    item_similarity = pairwise_distances(train_data_matrix.T, metric=metric)

    item_prediction = predict(train_data_matrix, item_similarity, type_='item')
    user_prediction = predict(train_data_matrix, user_similarity, type_='user')

    print '================'
    print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
    print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))

    return item_prediction, user_prediction


header = ['UserID','MovieID','Rating','Timestamp']
header_items = ['MovieID','Title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
genres = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

if __name__ == '__main__':

    #LOAD DATA - MOVIELENS
    df = pd.read_csv('ml-100k/u.data',sep='\t',names=header)
    n_users = df.UserID.unique().shape[0]
    n_items = df.MovieID.unique().shape[0]
    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

    movies_df = pd.read_csv('ml-100k/u.item',sep='|',names=header_items)
    del movies_df['video release date']
    del movies_df['IMDb URL']

    #Create two user-item matrices, one for training and another for testing
    train_data_matrix, test_data_matrix = traintestset(df,n_users,n_items,prop=0.25)

    ###### CF - Memory based #####
    item_prediction, user_prediction = cf_memory_based(train_data_matrix,test_data_matrix)

    #Recommend movies
    # print 'reeeeeeecoooooooo'
    # already_rated_item , item_pred = recommend_movies(item_prediction, 836, movies_df, df, test_data_matrix, 10, example = True)
    print 'reeeeeeecoooooooo'
    already_rated_user , user_pred = recommend_movies(user_prediction, 836, movies_df, df, test_data_matrix, 10)
    exit()

    print '================'
    print '================'

    ##### CF - Model based #####
    sparsity=round(1.0-len(df)/float(n_users*n_items),3)
    print 'The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%'
    #get SVD components from train matrix. Choose k.
    mean_user_rating = np.array([ np.mean([r for r in rates if r > 0]) for rates in train_data_matrix])
    train_data_matrix = train_data_matrix - mean_user_rating[:,np.newaxis]

    u, s, vt = svds(train_data_matrix, k = 20)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt) + mean_user_rating[:,np.newaxis]
    print 'User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix))

    print 'reeeeeeecoooooooo'
    already_rated, predictions = recommend_movies(X_pred, 836, movies_df, df, test_data_matrix, 10)
