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

[App](https://github.com/chrismartinezb/Game-of-thrones-analysis/blob/master/Final_Pipeline.ipynb): The whole process from data preparation and analysis to modeling and scoring.

[Data](https://github.com/chrismartinezb/Game-of-thrones-analysis/blob/master/GoT%20dialogue%20generator.ipynb):  ETL pipeline used to process data in preparation for model building.

[Models](https://github.com/chrismartinezb/Game-of-thrones-analysis/tree/master/GoT): All raw form of chapter scripts obtained from [Genius.com](https://genius.com/artists/Game-of-thrones).
 

## Licensing, Authors, Acknowledgements
I have to give credit to all the volunteers at Genius for transcribing the dialogues as well as to [Paras Chopra](https://towardsdatascience.com/generating-new-ideas-for-machine-learning-projects-through-machine-learning-ce3fee50ec2) and [Daniel E. Licht](https://lichtphyz.github.io/) whose work is the base of this project. Apart from that feel free to use the code and files as you wish.
