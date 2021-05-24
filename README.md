# Disaster Tweet Classification
This is a project to solve the problem stated in [Kaggle: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
I followed Google TensorFlow work [Classify text with BERT](https://www.tensorflow.org/tutorials/text/classify_text_with_bert), and eventually deployed the solution as a web api to Amazon AWS EC2. Used BERT NLP pre-trained model because it is relatively newer model for sentiment analysis with more sophisticated deep learning framework(attention, transfomer) that should provider higher accuracy. I have experimented with several models listed in the TensorFlow tutorial, and [wiki_books/sst2](https://tfhub.dev/google/experts/bert/wiki_books/sst2/2) fine-tuned for sentiment analysis performed around 82%(0.82071) accuracy on test dataset.


## To install
1. create local repository and pull from github
2. create virtualenv (recommended). python version used is 3.8.5
3. pip install -r requirements.txt

## To run
1. from project root directory: uvicorn tweet.api:app
2. use tools like curl or Postman to POST to http://localhost:8000/classify
  - body should be json 
  {
    "data" : [ "tweet1", "tweet2",....]
  }
  
  - response 200
  {
    [label1, label2, ...] #0:no disaster 1: disaster
  }
3. pre-trained and saved model should already be in tweet/model/saved directory

## project structure
/data : csv files for train, test and evaluation
  train.csv: training data
  test_labeled.csv: 100 manually labeled data for quick evaluation
  test.csv: test data for actual prediction

/explore: Jupyter notebooks for data exploration and testing

/tweet/model: model used in classification.

/tweet/test: testing units 

below log files will be created from where the code is run
api.log: logs for webapi 
ml.log: classifier model logs


Procedures on how to package and deploy on AWS EC2 is [here](https://github.com/op221/tweet_classification/blob/master/DEPLOYMENT.md)
