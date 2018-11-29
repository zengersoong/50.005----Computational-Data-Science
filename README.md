# 50.038 -- Computational Data Science

## Lab 1 
### Data handling
Instructor(s): Prof. Dorien Herremans  
1. Basic commands on unix
2. Scrapping with beautiful soup

## Lab 2
### Big data / Hadoop & MapReduce
Instructor(s): Prof. Dorien Herremans  
1. Hadoop filesystem commands
2. Map reduce
3. Coding map reduce for different task

## Lab 3
### Classification using Weka & Scikit-learn
Instructor(s): Prof. K.H.Lim
1. Weka, diabetes. training and testing classifiers
2. Randomtree and randomforest classifiers
3. Twitter sentiment datasets with scikit-learn
4. Count vectorizer function to count the frequency of each word
5. Training and testing classifier
6. Implement simple pipeline with Naive Baytes classifier with words
7. Removing stop-words to improve the model
8. Enhance the model by experimenting with bi-grams and tri-grams

## Lab 4
### Data visualisation in Python
Instructor(s): Prof. Dorien Herremans  
Using python pandas, as well as matplotlib to visualise data
1. Python pandas dataframes
2. Matplotlib library functions and plots
3. Histograms, bar charts, line plots , box plots , scatterplots
## Lab 5
### Feature Selection and Time Series
Instructor(s): Prof. K.H.Lim   
Using Sklearn machine learning library for data pre-processing, classification, clustering . 
Using Twitter datasets for sentiment labelling. 4=positive and 0 = negative

1. Machine learning through sklearn . 
2. Data preprocessing and classification . 
3. Using different feature selections to optimise the classifer/ improve results (top - k Features, kNearestNeighbours, top-k percentile) . 
4. Using Weka with package, "Time Series Forecasting" . (Only screenshots, in weka pdf)
5. Evaluate linear regression with / without attribute(feature) selection.  

## Lab 6
### Feature Selection and Time Series
Instructor(s): Prof. K.H.Lim   
Using Weka to explore the idea of data pre-processing, as well as using Association Rule Mining.  
#### Data Sets used are :  
1. credit-g.arff dataset
2. supermarket.arff
3. iris.arff
#### Association Rule Mining/Preprocessing
1. Using unsupervised discretizer to filter the attributes
2. Using Weka's Apriori algorithm for association rule Mining
3. Identifying the features
4. Apply minimum support threshold and identifying top 10 association rules
5. Assocation rules are ranked by their associated confidence score
#### Clustering
1. Using K-means with and without normalisation
2. Observing the effects on the SSE values (Normalized vs non-normalized)

## Lab 9
### Multiple-layer Perceptron
Instructor(s): Prof. K.H.Lim   
a. Multi-layer Perception for image classifcation  
b. Multi-layer Perception for text classifcation
#### Data Sets used are :  
1. MNIST dataset
2. Reuters news article Dataset
#### Data preprocessing (a)
1. Reshaping the data from 2D to 1D.
2. Cinverting to Float data type
3. Divide by 255 for pixel-shader value this normalizes to between 0-1.
4. Split the data to training set and testing set.
#### Neural Network model (a)
1. Set number of nodes as as well as hidden layers
2. Set input shape ( Number of features)
3. Adding a random dropout to avoid over-reliance on particular nodes.
#### Neural Network compile (a)
1. Loss = categorical crossentropy
2. Optimizer set to using adam
3. Metrics set to 'accuracy'
#### Fitting model to training set (a)
1. Setting Batch size (mini-batch training)
2. Set number of epoch(1 EPOCH = 1 forward propagation + 1 backward propagation)
3. Print the score of the accuracy of the model by evaluating with test set
4. Done

## Lab 10
### Word2Vec
Instructor(s): Prof. Dorien Herremans  
a. Training word2Vec from scratch  
b. Classification with word2Vec 
#### Repository Prof. Dorien Herremans 
1. https://github.com/dorienh/computational_data_science/blob/master/lab10a%20-%20word2vec:%20training%20a%20basic%20model%20from%20scratch.ipynb . 
2. https://github.com/dorienh/computational_data_science/blob/master/lab10b%20-%20word2vec%20classification.ipynb

# Lab 12
### Recurrent Neural Networks for text classification
Instructor(s): Prof. K.H.Lim  
Date: 29/11/18
a. Text Classification
b. Generate Text

# Text Classification
1. Using either Simple RNN / LSTM / GRU model for the model to be trained on
2. Test the output against the actual for accuracy

# Text Generation
1. Preprocessing
2. Mapping characters to integer. Text generation will generate char by char
3. Prepare datasets, using sequences of 50 char to generate 51st char
4. Construct LSTM model
5. Generate a new text based on provided seed text.

