import numpy as np
import pandas as pd
import re, nltk, string
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle


def get_recommendations(user_input):
    url = 'https://raw.githubusercontent.com/adwayskirwe/Capstone/main/sample30.csv'
    reviews_df = pd.read_csv(url)

    reviews_df['user_sentiment_bool'] = reviews_df['user_sentiment'].apply(lambda x: 0 if x == "Negative" else 1)
    reviews_df['reviews_username'].fillna('others',inplace=True)
    reviews_df['user_sentiment'].fillna('Positive',inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(reviews_df,reviews_df['user_sentiment_bool'],test_size=0.3,shuffle=False)



    ############################################################################################################################################

    wordnet_lemmatizer = WordNetLemmatizer()

    def preprocess(document):
        'changes document to lower case and removes stopwords, punctuation, numbers and convert words to root form using wordnet_lemmatizer'

        # Make the text lowercase
        document = document.lower()
        
        #Remove punctuation and words containing numbers
        document = re.sub("[^\sA-z]","",document)
        
        # tokenize into words
        words = word_tokenize(document)
        
        # remove stop words
        words = [word for word in words if word not in stopwords.words("english")]
        
        # Lemmatizing the words
        words = [wordnet_lemmatizer.lemmatize(word) for word in words]
        
        # join words to make sentence
        document = " ".join(words)
        
        return document


    '''
    preprocessed_review = [preprocess(review) for review in tqdm(X_train['reviews_text'])]
    X_train['preprocessed_review'] = pd.Series(preprocessed_review)

    preprocessed_review = [preprocess(review) for review in tqdm(X_test['reviews_text'])]
    X_test['preprocessed_review'] = preprocessed_review


    vectorizer = TfidfVectorizer(max_df=0.95,min_df=3)
    X = vectorizer.fit_transform(X_train['preprocessed_review'])
    features_df = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())


    X_test_ = vectorizer.transform(X_test['preprocessed_review'])
    features_df_test = pd.DataFrame(X_test_.toarray(), columns = vectorizer.get_feature_names())




    ############################################################################################################################################

    logisticRegression = LogisticRegression(random_state=100)
    logisticRegression.fit(features_df,y_train)

    y_test_pred = logisticRegression.predict(features_df_test)
    logistic_regression_test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    print("logistic_regression_test_accuracy=",logistic_regression_test_accuracy)
    print("logistic_regression_f1_score =", f1_score(y_test, y_test_pred,average='weighted'))


    ############################################################################################################################################
    

    with open('model_pkl', 'wb') as files:
        pickle.dump(logisticRegression, files)
    '''


    sample_df = reviews_df[['reviews_username','name','reviews_rating']]
    sample_df_groupby = sample_df.groupby(['reviews_username','name']).mean()
    sample_df_groupby = sample_df_groupby.reset_index()

    train, test = train_test_split(sample_df_groupby, test_size=0.30, random_state=31)

    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(0)


    dummy_train = train.copy()

    # The movies not rated by user is marked as 1 for prediction. 
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(1)


    from sklearn.metrics.pairwise import pairwise_distances

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0


    # Create a user-product matrix.
    df_pivot = train.pivot(
       index='reviews_username',
       columns='name',
       values='reviews_rating'
    )


    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T-mean).T


    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0


    user_correlation[user_correlation<0]=0


    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))


    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)


    # Take the user ID as input.
    #user_input = input("Enter your user name")
    print("User name input = ", user_input)


    top20_recommendations_for_user = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Find out the common users of test and train dataset.
    common = test[test.reviews_username.isin(train.reviews_username)]


    # convert into the user-product matrix.
    common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


    # Convert the user_correlation matrix into dataframe.
    user_correlation_df = pd.DataFrame(user_correlation)


    user_correlation_df['reviews_username'] = df_subtracted.index
    user_correlation_df.set_index('reviews_username',inplace=True)


    list_name = common.reviews_username.tolist()
    user_correlation_df.columns = df_subtracted.index.tolist()
    user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]
    user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]
    user_correlation_df_3 = user_correlation_df_2.T


    user_correlation_df_3[user_correlation_df_3<0]=0

    common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))


    dummy_test = common.copy()

    dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

    dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)


    common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)

    '''
    from sklearn.preprocessing import MinMaxScaler
    #from numpy import *

    X  = common_user_predicted_ratings.copy() 
    X = X[X>0]

    scaler = MinMaxScaler(feature_range=(1, 5))
    scaler.fit(X)
    y = (scaler.transform(X))


    common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')

    # Finding total non-NaN value
    total_non_nan = np.count_nonzero(~np.isnan(y))

    #rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
    #print("RMSE (Root Mean Squared Error = ", rmse)
    '''

    ############################################################################################################################################


    with open('model_pkl' , 'rb') as f:
        logisticRegression = pickle.load(f)

    #List that will store postive-sentiment ratio for each of 20 recommended products
    positive_sentiment_ratio_list = []
    product_list = []

    #For each of the 20 recommended products 
    for i in range(0,len(top20_recommendations_for_user.index.tolist())):
      
    	#Get product name
        product = top20_recommendations_for_user.index.tolist()[i]

    	#Find out all postive + negative reviews about product from training dataset
    	#all_reviews_for_respective_product_df = reviews_df[reviews_df['name'] == product]['reviews_text']
        all_reviews_for_respective_product_df = reviews_df[reviews_df['name'] == product]['user_sentiment_bool']
    	  
    	#Preprocess all the above extracted reviews before predicting user sentiment
    	#preprocessed_review = [preprocess(review) for review in tqdm(all_reviews_for_respective_product_df)]
    	#all_reviews_for_respective_product_df = pd.Series(preprocessed_review)

    	#Creating TF-IDF strcture for all the reviews
    	#X = vectorizer.transform(all_reviews_for_respective_product_df)
    	#all_reviews_for_respective_product_features_df = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())

    	#Predict the sentiment of all the reviews of the product
    	#product_review_pred = logisticRegression.predict(all_reviews_for_respective_product_features_df)
    	  
    	#Find Positive sentiment ratio of the product 
        positive_sentiment_ratio = round(np.count_nonzero(all_reviews_for_respective_product_df == 1)/(np.count_nonzero(all_reviews_for_respective_product_df == 0)+np.count_nonzero(all_reviews_for_respective_product_df == 1)),2)

    	#Append the Positive sentiment ratio of the product to the list
        positive_sentiment_ratio_list.append(positive_sentiment_ratio)
        product_list.append(product)



    index_list = np.argsort(positive_sentiment_ratio_list)[-5:]


    product_name_list = []
    for i in range(0, len(index_list)):
        product_name = top20_recommendations_for_user.index.tolist()[index_list[i]]
        product_name_list.append(product_name)


    
    return product_name_list



