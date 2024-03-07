import os
import re
import gensim
import string
import json
import random
import numpy as np
import scipy.sparse
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import csv
from sklearn import preprocessing
from scipy.special import softmax
from scipy.sparse import csr_matrix
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
import pandas as pd
import plotly.express as px

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)


def preprocess_text(text):
    # Remove \n and urls
    text = text.replace('\n', ' ')
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Convert text to lower
    text = text.lower()

    # Remove Punctuation
    tokenizer = RegexpTokenizer(r'\w+\'?\w+|\w+')
    words = tokenizer.tokenize(text)
    words_without_punctuation = [''.join(c for c in word if c not in string.punctuation or c in ["'", "’"]) for
                                 word in words]
    text = ' '.join(words_without_punctuation)

    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmaWords = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
    text = ' '.join(lemmaWords)

    # Remove stop words
    stop_words_lower = set(word.lower() for word in stopwords.words('english'))
    stop_words_upper = set(word.title() for word in stopwords.words('english'))
    stop_words = stop_words_lower.union(stop_words_upper)
    tokens = word_tokenize(text)
    tokensNoSWs = [tok for tok in tokens if tok not in stop_words]
    processedText = ' '.join(tokensNoSWs)

    # Remove numbers
    processedText = re.sub(r'\d+', '', processedText)

    # Remove unimportant and common words, and singular letters from the text
    processedText = re.sub(r'\b\w\b', '', processedText)
    processedText = processedText.replace("'", "")

    return processedText


# Function sgd_for_multinomial_lr_with_ce is from class
def sgd_for_multinomial_lr_with_ce(X, y, num_passes=5, learning_rate=0.1):
    num_data_points = X.shape[0]
    num_classes = len(set(y))

    # Initialize theta -> 0
    num_inputs = X.shape[1]
    w = np.zeros((num_inputs, num_classes))
    b = np.zeros(num_classes)

    for current_pass in range(num_passes):

        # iterate through entire dataset in random order
        order = list(range(num_data_points))
        random.shuffle(order)
        for i in order:
            # compute y-hat for this value of i given y_i and x_i
            x_i = X[i]
            y_i = y[i]

            # need to use this instead of y_i!
            y_i_onehot = [0] * num_classes
            y_i_onehot[y_i] = 1

            # need to compute based on w and b
            # sigmoid(w dot x + b)

            z = x_i.dot(w) + b
            y_hat_i = softmax(z)

            # for each w (and b), modify by -lr * (y_hat_i - y_i) * x_i
            w = w - learning_rate * ((y_hat_i - y_i_onehot).T @ x_i).T
            b = b - learning_rate * (y_hat_i - y_i_onehot)

    # return theta
    return w, b


# Function make_predictions_multinomial is from class too
def make_predictions_multinomial(w, b, X):
    outputs = X.dot(w) + b
    return np.argmax(outputs, axis=1)


def sentence_vector(sentence, model):
    vector_size = model.vector_size
    result = np.zeros(vector_size)
    ctr = 1
    for word in sentence:
        if word in model:
            result += model[word]
            ctr += 1
    result = result / ctr
    return result


# Convert gensim model to WEFE model
def convert_gensim_to_wefe_model(gensim_model):
    # Check if the model is a Word2Vec model
    if isinstance(gensim_model, gensim.models.Word2Vec):
        wefe_model = WordEmbeddingModel(gensim_model.wv)
    # If the model is not a Word2Vec model, it's assumed to be a KeyedVectors object
    else:
        wefe_model = WordEmbeddingModel(gensim_model)
    return wefe_model


def homeworkTwo():
    # 2.1 Dataset

    if not os.path.exists("processed_sentences.json"):
        texts = dataset['train']['text']
        processed_sentences = [];
        for text in texts:
            processed_text = preprocess_text(text)
            processed_sentences.append(processed_text)
        with open('processed_sentences.json', 'w') as f:
            json.dump(processed_sentences, f)

    # Open the file with the preprocessed dataset
    with open('processed_sentences.json', 'r') as f:
        processed_sentences = json.load(f)
    tokenized_sentences = [sentence.split() for sentence in processed_sentences]

    # 2.2 Training Word Embeddings

    # Create CBOW model
    if not os.path.exists("cbow_model.model"):
        cbowModel = gensim.models.Word2Vec(tokenized_sentences, min_count=1, vector_size=200, window=5, workers=8)
        cbowModel.save("cbow_model.model")

    # Create Skip Gram model
    if not os.path.exists("skip_gram_model.model"):
        skipGramModel = gensim.models.Word2Vec(tokenized_sentences, min_count=1, vector_size=200, window=5, sg=1,
                                               workers=8)
        skipGramModel.save("skip_gram_model.model")

    # Load the models from the files
    cbowModel = gensim.models.Word2Vec.load("cbow_model.model")
    skipGramModel = gensim.models.Word2Vec.load("skip_gram_model.model")

    # 2.3 Comparing Word Embeddings
    googleNews_model = api.load('word2vec-google-news-300')
    glove_model = api.load('glove-wiki-gigaword-200')

    # Query One: Finding the 5 most similar words to "hydrogen" in all four models
    print("Query One...")
    print("The 5 most similar words to 'hydrogen' in CBOW model: ", cbowModel.wv.most_similar('hydrogen', topn=5))
    print("The 5 most similar words to 'hydrogen' in Skip Gram model: ",
          skipGramModel.wv.most_similar('hydrogen', topn=5))
    print("The 5 most similar words to 'hydrogen' in Google News model: ",
          googleNews_model.most_similar('hydrogen', topn=5))
    print("The 5 most similar words to 'hydrogen' in Glove model: ", glove_model.most_similar('hydrogen', topn=5))
    print("\n")

    # Query Two: Finding the 5 most similar words to "michigan" in all four models
    print("Query Two...")
    print("The 5 most similar words to 'michigan' in CBOW model: ", cbowModel.wv.most_similar('michigan', topn=5))
    print("The 5 most similar words to 'michigan' in Skip Gram model: ",
          skipGramModel.wv.most_similar('michigan', topn=5))
    print("The 5 most similar words to 'michigan' in Google News model: ",
          googleNews_model.most_similar('michigan', topn=5))
    print("The 5 most similar words to 'michigan' in Glove model: ", glove_model.most_similar('michigan', topn=5))
    print("\n")

    # Query Three: Finding the similarity of two opposite words
    print("Query Three...")
    print("The similarity between 'hot' and 'cold' in CBOW model: ", cbowModel.wv.similarity('hot', 'cold'))
    print("The similarity between 'hot' and 'cold' in Skip Gram model: ", skipGramModel.wv.similarity('hot', 'cold'))
    print("The similarity between 'hot' and 'cold' in Google News model: ", googleNews_model.similarity('hot', 'cold'))
    print("The similarity between 'hot' and 'cold' in Glove model: ", glove_model.similarity('hot', 'cold'))
    print("\n")

    # Query Four: Finding the similarity of two similar words
    print("Query Four...")
    print("The similarity between 'hot' and 'warm' in CBOW model: ", cbowModel.wv.similarity('hot', 'warm'))
    print("The similarity between 'hot' and 'warm' in Skip Gram model: ", skipGramModel.wv.similarity('hot', 'warm'))
    print("The similarity between 'hot' and 'warm' in Google News model: ", googleNews_model.similarity('hot', 'warm'))
    print("The similarity between 'hot' and 'warm' in Glove model: ", glove_model.similarity('hot', 'warm'))
    print("\n")

    # Query Five: Checking if the model can pick the odd one out
    print("Query Five...")
    odd_cbow = cbowModel.wv.doesnt_match(['football', 'baseball', 'basketball', 'car'])
    print("The odd one out in the list ['football', 'baseball', 'basketball', 'car'] in CBOW model: ", odd_cbow)
    odd_sg = skipGramModel.wv.doesnt_match(['football', 'baseball', 'basketball', 'car'])
    print("The odd one out in the list ['football', 'baseball', 'basketball', 'car'] in Skip Gram model: ", odd_sg)
    odd_wv = googleNews_model.doesnt_match(['football', 'baseball', 'basketball', 'car'])
    print("The odd one out in the list ['football', 'baseball', 'basketball', 'car'] in Google News model: ", odd_wv)
    odd_glove = glove_model.doesnt_match(['football', 'baseball', 'basketball', 'car'])
    print("The odd one out in the list ['football', 'baseball', 'basketball', 'car'] in Glove model: ", odd_glove)
    print("\n")

    # 2.4 Bias in Word Embeddings
    # Create two queries
    RNSB_words = [["swedish"], ["irish"], ["mexican"], ["chinese"], ["filipino"], ["german"], ["english"], ["french"],
                  ["norwegian"], ["american"], ["indian"], ["dutch"], ["russian"], ["scottish"], ["italian"]]
    my_words = [["woman"], ["man"], ["mrs"], ["mr"], ["queen"], ["king"], ["nurse"], ["surgeon"], ["teacher"],
                ["engineer"], ["librarian"], ["scientist"]]
    bing_liu = load_bingliu()
    print("Creating a query for the RNSB_words set with nationalities...")
    query = Query(RNSB_words, [bing_liu["positive_words"], bing_liu["negative_words"]])
    print("Creating a query for the my_words set with careers commonly associated with one gender...")
    query2 = Query(my_words, [bing_liu["positive_words"], bing_liu["negative_words"]])
    print("\n")

    # Create a list of the models and their names
    models = [(cbowModel, 'CBOW'), (skipGramModel, 'Skip Gram'), (googleNews_model, 'Google News'),
              (glove_model, 'GloVe')]
    queries = [query, query2]

    # Loop over the models and their names
    for model, model_name in models:
        # Convert the model to a WEFE model
        model_wefe = convert_gensim_to_wefe_model(model)

        # Loop over the queries
        for i, query in enumerate(queries, 1):
            # Run the RNSB query for the model
            result = RNSB().run_query(query, model_wefe, lost_vocabulary_threshold=0.35)

            # Create a DataFrame from the result
            df_negative = pd.DataFrame(list(result['negative_sentiment_distribution'].items()),
                                       columns=['word', 'negative_sentiment_distribution'])

            # Plot the results
            fig = px.bar(df_negative, x='word', y='negative_sentiment_distribution',
                         title=f"Negative Sentiment Distribution for {model_name}",
                         labels={"negative_sentiment_distribution": "Negative Sentiment Distribution", "word": "Word"})
            fig.update_yaxes(range=[0, 0.2])
            fig.show()

    # 2.5 Text Classification

    # The code below is from class
    texts = []
    labels = []
    with open("Tweets.csv") as infile:
        for line_dict in csv.DictReader(infile):
            texts.append(line_dict['text'])
            labels.append(line_dict['airline_sentiment'])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)
    le.inverse_transform([0, 1, 2])
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    X = vectorizer.fit_transform(texts)
    softmax([1, 2, 3])
    m_w, m_b = sgd_for_multinomial_lr_with_ce(X, y)
    outputs = X.dot(m_w)
    preds = np.argmax(outputs, axis=1)
    preds = make_predictions_multinomial(m_w, m_b, X)
    print("Printing out classification report for the text classification with bag of words features...")
    print(classification_report(y, preds))
    print("\n")

    # Train a second linear regression model for only one word embedding
    sentence_vectors = []
    for sentence in texts:
        sentence = preprocess_text(sentence)
        sentence_vectors.append(sentence_vector(sentence, googleNews_model))
    X = np.array(sentence_vectors)
    X = scipy.sparse.csr_matrix(X)
    m_w, m_b = sgd_for_multinomial_lr_with_ce(X, y)
    outputs = X.dot(m_w)
    preds = np.argmax(outputs, axis=1)
    preds = make_predictions_multinomial(m_w, m_b, X)
    print("Printing out classification report for the text classification with cbow features...")
    print(classification_report(y, preds))


if __name__ == "__main__":
    homeworkTwo()
