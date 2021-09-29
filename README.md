<meta name="robots" content="noindex">

# NLPSentimentAnalysis
### Building	a	sentiment	classifier	for	Twitter	
#### MSc Computer Science. Natural Language Processing. Sentiment Analysis 

#### Preprocessing

As part of the classification task, the	tweets are preprocessed by applying various NLP techniques.

The following steps have been performed during preprocessing:
1. Replacing emojis, expressing happiness, with the word “posemojy” :) :] :3 :> 8) (: =) =] :’) :-)
2. Replacing emojis, expressing sadness, with the word “negemojy” :(:[:<8( ):=(=[:’(:-(
3. Replacing user mentions (with @ at the beginning) with the word “usermention” 4. Replacing laughing e.g. haha, ahaha, lol... with the word “laughintweet”
5. Replacing urls with the word “urlintweet”
6. Deleting all the numbers and words that include numbers
7. Changing elongated words to no more than 2 same letters (for example, Hmmmm -> Hmm) 8. Deleting non-alphanumeric characters
9. Deleting one-letter words
10. Replacing words, expressing negations, with a special word “negationintweet”.
11. All words were converted into lower case.

For tokenisation, TweetTokenizer was used. Words written with non-English symbols were ignored.
WordNetLemmatizer was used for reducing the size of vocabulary. Before it, POS tagging (nltk.tag.pos_tag) was performed to help the lemmatizer. For the purposes of this notebook only nouns and verbs were given their non-default tags. All other words were considered as adjectives.


#### Feature extraction

For feature extraction both CountVectorizer and TFIDF from sklearn were used separately. By this, I divided my work into two parts. In both cases, after embedding, the matrices were reduced by applying Χ2 testing (chi-squared testing) in order to find 500 important features. This number of features was chosen after cross-validation.
Three additional features were added: the number of positive words, the number of negative words and the number of bad words occurred in each tweet. The presence of these types of words could be highly correlated with the target label. The sets of positive and negative words were downloaded from https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html. The set of bad words - https://www.cs.cmu.edu/~biglou/resources/. The dimension of the final feature vector is 503. 500 words chosen by chi-squared testing and 3 features - “PositiveWords”,”NegativeWords”,”BadWords”.


#### Classifiers

In this notebook two families of machine learning algorithms are used: (1) traditional machine learning methods such as Multinomial Naive Bayes, Gaussian Naive Bayes and Passive Aggressive Classifier trained on different sets of features; (2) long short term memory (LSTM) neural networks evaluated using Keras library.

#### Glove Word Embedding

glove.6B.100d.txt contains a 100-dimensional version of the embedding. It presents a word followed by the weights (100 numbers) on each line. Keras provides a Tokenizer class that can be fit on the training data, converts text to sequences (texts_to_sequences() method) and provides
a word_index attribute.
Categorical cross-entropy was used as loss with the rmsprop optimizer. After testing the softmax, tanh, and sigmoid activation functions, I found that sigmoid provided the best results.
The model layers can be presented as following:
Embedding(5000, 100) -> LSTM(100) -> Dense(3, activation = sigmoid)

#### Use

To evaluate my code on additional data, the file locations of testsets should be entered into the list “filenames”. Then, the whole notebook should be run, with the results being output in the final two cells.
Additional files negative-words.txt, positive-words, bad-words are added to the folder. 
