# Disaster response pipeline

The aim of the project is to build a Natural Language Processing tool that categorize messages in order to prioritize the important ones
during a real life desaster event. 

# Install

This project requires **Python 3.6** and the following Python libraries installed:

- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [nltk](https://www.nltk.org/install.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Sqlalchemy](https://www.sqlalchemy.org/)
- [Flask](http://flask.pocoo.org/)
- [Plotly](https://plot.ly/)

## Data

The initial dataset contains 26,216 pre-labelled tweets and messages from real-life disaster coutesy of [Figure Eight](https://www.figure-eight.com/). They fall in one or several of the next 36 categories:

'related',
 'request',
 'offer',
 'aid_related',
 'medical_help',
 'medical_products',
 'search_and_rescue',
 'security',
 'military',
 'child_alone',
 'water',
 'food',
 'shelter',
 'clothing',
 'money',
 'missing_people',
 'refugees',
 'death',
 'other_aid',
 'infrastructure_related',
 'transport',
 'buildings',
 'electricity',
 'tools',
 'hospitals',
 'shops',
 'aid_centers',
 'other_infrastructure',
 'weather_related',
 'floods',
 'storm',
 'fire',
 'earthquake',
 'cold',
 'other_weather',
 'direct_report'
 
 ## Instructions 

1. Run the following commands in the project's root directory to set up your database and model:
- To run ETL pipeline that cleans data and stores in database `data/disaster_messages.csv data/disaster_categories.csv` data/DisasterResponse.db

- To run ML pipeline that trains classifier and saves python `models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2.Run the following command in the app's directory to run your web app. `python run.py`

3. Go to `http://0.0.0.0:3001/`



## File description.

[app](https://github.com/chrismartinezb/Disaster_response_pipeline/tree/master/app): HTML templates for the web app and run.py that starts the webapp.

[data](https://github.com/chrismartinezb/Disaster_response_pipeline/tree/master/data):  ETL pipeline used to process data in preparation for model building.

[models](https://github.com/chrismartinezb/Disaster_response_pipeline/tree/master/models): The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle
 

## Licensing, Authors, Acknowledgements

- To [Udacity](https://www.udacity.com/) for providing the starter code as part of their Data Sciencde Nanodegree program.

- To  [Figure Eight](https://www.figure-eight.com/) for providing the dataset.

For the remaining code, feel free to use it as you wish.
