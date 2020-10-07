# -*-coding:Latin-1 -*

# Machine Learning 2 Homework 1
print("\nMachine Learning 2 Homework 1")
# By Kutlu Toren & Louis Devèze
print("By Kutlu Toren & Louis Deveze")
# Implementation of a Spam filter using Naive Bayes implementation
print("Implementation of a Spam filter using Naive Bayes implementation\n")

# --------------------
# READ ME
# To run the program just enter the command line python NaiveBayes.py 
# The output of the program is also available inside the Output.txt file
# The convention choosed is SPAM = 1, HAM = 0


# --------------------
# Imports definitions
from collections import Counter
import numpy

# --------------------
# Constants definitions
tr_ratio = 0.75
word_amount = 2000
alpha = 2

# --------------------
# Function definitions

# Create a dictionary of the most common words used inside the messages
# given as arguments
def create_word_dictionary(messages): 
    # List of words
    words = []
    for message in messages:
        
        w = message.split()
        for word in w:
            # Adding only word with a length > 1
            if word != "spam" and word != "ham" and len(word) > 1:
                words.append(word)
    # Create the dictionnary out of the words
    dictionary = Counter(words)
    dictionary = dictionary.most_common(word_amount)
    return dictionary

# Create a matrix of features
# Each row represents a message
# Each colums represents a dictionary word
# Each cell is either one if the message contains this word or zero
def create_word_matrix(messages, dictionary):
    features = numpy.zeros((len(messages), len(dictionary)))

    # Iterate over messages
    messageID = 0
    for message in messages:
        
        # Iterate over Words
        wordID = 0
        for word, count in dictionary:

            # If the message contains the word
            if message.find(word) != -1:
                features[messageID, wordID] = 1
            
            wordID = wordID+1
        
        messageID = messageID + 1

    return features

# Create a vector containing the label SPAM (1) or HAM (0) for evey message given as argument
def create_label_vector(messages):
    labels = numpy.zeros(len(messages))

    labelID = 0
    for message in messages:

        words = message.split()
        # mark spams message with a 1
        if words[0] == "spam":
            labels[labelID] = 1

        labelID += 1

    #return the vector of labels    
    return labels

# Compute the vector of the probability for each word to appear into a SPAM
def train_spam_vector(labels, features):

    # Create the Spam probability vector
    word_spam_probabilities = numpy.zeros(word_amount)

    spam_count = 0

    # First check the occurence of each word in spam messages of the train set
    # For each Spam Message
    for row in range(0, len(labels)):
        if labels[row] == 1:
            spam_count += 1
            # Add each word prob the features (1 if contained, 0 if not)
            for col in range(0, word_amount):
                word_spam_probabilities[col] += features[row, col]

    # Divide the occurences by the amount of spam + Laplacian smoothing
    for col in range(0, word_amount):  
        word_spam_probabilities[col] += 1
        word_spam_probabilities[col] /= (spam_count +  (2 * alpha))

    
    print("Training sample has ", spam_count, " SPAMs")

    return word_spam_probabilities

# Compute the vector of the probability for each word to appear into a HAM
def train_ham_vector(labels, features):

    # Create the Ham probability vector
    word_ham_probabilities = numpy.zeros(word_amount)

    ham_count = 0

    # First check the occurence of each word in Ham messages of the train set
    # For each Ham Message
    for row in range(0, len(labels)):
        if labels[row] == 0:   
            ham_count += 1
            # Add each word prob the features (1 if contained, 0 if not)
            for col in range(0, word_amount):    
                word_ham_probabilities[col] += features[row, col]

    # Divide the occurences by the amount of ham  + Laplacian smoothing
    for col in range(0, word_amount):  
        word_ham_probabilities[col] += alpha
        word_ham_probabilities[col] /= (ham_count + (2 * alpha))

    
    print("Training sample has ", ham_count, " HAMs")

    return word_ham_probabilities

# Compute the vector of the probability for each word to appear into a HAM
def train_word_prob(labels, features):

    # Create the Ham probability vector
    word_prob = numpy.zeros(word_amount)

    # First check the occurence of each word in messages of the train set
    # For each Message
    for row in range(0, len(labels)):
        # Add each word prob the features (1 if contained, 0 if not)
        for col in range(0, word_amount):    
            word_prob[col] += features[row, col]

    # Divide the occurences by the amount of labels
    for col in range(0, word_amount):  
        word_prob[col] /= len(labels)

    return word_prob

# Useful function to create a confusion matrix using the real output
def confusion_matrix(predicted, current):

    amount = len(predicted)

    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    for i in range(0, amount):
        if predicted[i] == current[i] and predicted[i] == 1:
            true_positive += 1
        elif predicted[i] == current[i] and predicted[i] == 0:
            true_negative += 1
        elif predicted[i] != current[i] and predicted[i] == 1:
            false_positive += 1
        elif predicted[i] != current[i] and predicted[i] == 0:
            false_negative += 1

    true_positive = true_positive / amount * 100
    true_negative = true_negative / amount * 100
    false_positive = false_positive / amount * 100
    false_negative = false_negative / amount * 100
    accuracy = true_positive + true_negative
    
    print("Set with a size of ", len(predicted))
    print("Global accuracy ", accuracy , "%")
    print("True positive ", true_positive, "%")
    print("True negative ", true_negative, "%")
    print("False positive ", false_positive, "%")
    print("False negative ", false_negative, "%\n")

# Compute a vector of predicted output for the given input messages
def predict(spam_model, ham_model, word_model, dictionary, messages):

    # Create an array containing the messages output
    output = numpy.zeros(len(messages))

    spam_proba = numpy.zeros(word_amount)
    ham_proba = numpy.zeros(word_amount)

    # create the Test features matrix
    features = create_word_matrix(messages, dictionary)

    # For each Message
    for row in range(0, len(messages)):
        for col in range(0, word_amount):    
            if features[row, col] == 1:
                # Computing P(ϕn|Y=1)*P(ϕY)
                spam_proba[col] = spam_model[col] * word_model[col]
                # Computing P(ϕn|Y=0)*P(ϕY)
                ham_proba[col] = ham_model[col] * word_model[col]
            else:
                spam_proba[col] = 0
                ham_proba[col] = 0
        
        # Now that spam proba and ham_proba are computed calculate output prediction
        spam = spam_proba.sum()
        spam = spam
        ham = ham_proba.sum()
        if spam > ham:
            output[row] = 1
        else:
            output[row] = 0

    return output


# --------------------
# I - Data Loading
# Divide the data in two groups: training and test examples.

# Getting the number of messages
messages_number = len(open('messages.txt').readlines())
# Getting the number of training messages
tr_amount = messages_number * tr_ratio
te_amount = int(tr_amount)
# Getting the number of test messages
te_amount = messages_number - tr_amount
test = []
train = []

# File opening and reading of the text
i = 1
with open("messages.txt") as f:
    for line in f:
        if i <= tr_amount:
            train.append(line)
        else:
            test.append(line)   
        i += 1

# Print training & test messages amounts
print('Training Set Amount: ', len(train), '  | Test Set Amount: ', len(test))

#------------------
# Data Model Creation
# II - Parse both the training and test examples to generate both the spam and ham data sets.
# III - Generate a dictionary from the training data.
# IV - Extract features from both the training data and test data.


# create the dictionary of features
dictionary = create_word_dictionary(train)
# create the features matrix
features = create_word_matrix(train, dictionary)
# create the label vector
labels = create_label_vector(train)

#------------------
# Model Training

# create the vector ϕn|Y=1
# it represents the probability for each word to appear into a spam
spam_model = train_spam_vector(labels, features)

# create the vector ϕn|Y=0
# it represents the probability for each word to appear into a ham
ham_model = train_ham_vector(labels, features)

# create the vector ϕY
# it represents the probability for each word to appear
word_model = train_word_prob(labels, features)

#------------------
# Model Prediction on Test Set
# V - Implement the Naive Bayes from scratch, ﬁt the respective models to the training data

# create the test label vector
test_labels = create_label_vector(test)

# VI - Make predictions for the test data.
# Calculate for each message a predicted output for spam on test Set
test_predicted = predict(spam_model, ham_model, word_model, dictionary, test)

# VII - Measure the spam-ﬁltering performance for each approach through the confusion matrix.
print("\nConfusion Matrix for Test" , end =" " )
confusion_matrix(test_predicted, test_labels)

#------------------
# Model Prediction on Train Set

# Calculate for each message a predicted output for spam on test Set
train_predicted = predict(spam_model, ham_model, word_model, dictionary, train)

print("Confusion Matrix for Train" , end =" " )
confusion_matrix(train_predicted, labels)

# VIII - Discuss your results.
# We provided the output of our homework into the Output.txt

# After few tweeks among the parameters and few modifications on the code parsing the data,
# We reached a pretty good accuracy result of 83,6 % (test set) and 84.4 % (train set).
# We noticed that the False positive ratio is pretty high on our model, compared to the
# false negative ratio which is less than 1%. At first, False Positive was about 25% percent 
# of the global output. We reduced it below 15% by allowing words with numerical values,
#  which often appear in some spams when we looked into the dataset. 

# On the second hand we noticed that the dictionnary is more efficient with a size around 2000 - 2500 words
# Otherwise, many low frequent words appear in it. It appears to have a negative impact on the score

# Finally we tried to add the Laplacian smoothing when training the model, which stabilizes the accurracy around 85.3 Percent
