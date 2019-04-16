import sys
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier

def load_data(database_filepath):
    '''
    Load data from database and split it into target and features
    Input:
        database_filepath: File path of sql database
    Output:
        X: features
        y: target
        category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster', engine)
    X = df.message
    y = df.drop(["message","id","original", "genre"], axis = 1)
    category_names = list(y)

    return X, y, category_names


def tokenize(text):
    ''' tokenizes and lemmatizes the message text 
    
    Input: Raw message text from dataframe
    
    Output: Cleaned tokenized and lemmatized text
    '''
    
    tokens = word_tokenize(text)
    
    
    lemmatizer = WordNetLemmatizer()

    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    ''' Builds a multiuotuput classifier Pipeline with 
        XGBoostClassifier and tfidf.
        
        Input: None
        
        Output: GridSearch Object.
       ''' 
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])
    
    parameters = {
        'clf__estimator__gamma': [0.5, 1, 1.5]
        }
    
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model with f1 score, recall, presicion and accuracy as metric.
    Input: 
        model: Model 
        X_test: features
        y_test: True lables 
        category_names: Labels for 36 message categories 
    Output:
        table with score for each categorie
    '''
    Y_pred = model.predict(X_test)
    
    # Calculates the accuracy
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
    


def save_model(model, model_filepath):
    '''
    Saves model as a pickle file 
    Input: 
        model: Model 
        model_filepath: path to the file
    Output:
        A pickle file of model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()