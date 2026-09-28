import pandas as pd
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
import matplotlib.pyplot as plt
from custom_stop_words_pattern import my_stop_words
from psychology_data import prepare_psychology_data


#params for coherence test
# Set DO_COHERENCE_TEST to True if you want to run a coherence test to determine the optimal number of topics (useful for finding the best model). Set it to False if you want to create a model with the specified number of topics.
DO_COHERENCE_TEST = False
START = 5
LIMIT = 31
STEP = 5

# model settings
NUM_TOPICS = 10
PASSES = 10
WORKERS = 8

# data settings
USE_SUBSET = False
VERBOSE = True
if DO_COHERENCE_TEST: VERBOSE = False
TRUNCATE_SIZE = 50000
INSPECT_ROW = 10
csv_file = 'E:\\topic_model\\gc_dissertations_combined_v2.csv'
headline_column = 'abstract'

# dictonary settings
MAX_WORDS = 100000
MIN_WORDS = 15
MAX_PERCENTAGE = 0.5


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in my_stop_words and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def compute_coherence_values(dictionary, corpus, texts, start=2, limit=40, step=6):
    coherence_values = []
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.LdaMulticore(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=10,
            workers=1,
        )
        cm = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(cm.get_coherence())
    return coherence_values

if __name__ == '__main__':
    np.random.seed(2026)

    # load wordnet and create stemmer
    nltk.download('wordnet')
    stemmer = SnowballStemmer("english")

    # load data

    data = pd.read_csv(csv_file)
    # calling processed CSV with targeted program and year range from psychology_data.py
    data = prepare_psychology_data(data, headline_column)
    data_text = data[[headline_column]]

    data_text['index'] = data_text.index

    # make our dataset smaller for testing
    if USE_SUBSET:
        documents = data_text.truncate(before=1, after=TRUNCATE_SIZE)
    else:
        documents = data_text

    if VERBOSE:
        # verify the data looks correct
        print("\n\ndocument length is ", len(documents))
        print(documents[:5])
        # look at specific entry
        doc_sample = documents[documents['index'] == INSPECT_ROW].values[0][0]
        print(f'\n\nunedited row #{INSPECT_ROW}: ')
        words = []
        for word in doc_sample.split(' '):
            words.append(word)
        print(words)
        print(f'\n\n tokenized and lemmatized row #{INSPECT_ROW}: ')
        print(preprocess(doc_sample))

    # preprocess all the docs
    processed_docs = documents[headline_column].map(preprocess)

    if VERBOSE:
        print("\n\npreprocessed docs:")
        print(processed_docs[:10])

    # Create dictionary
    dictionary = gensim.corpora.Dictionary(processed_docs)

    if VERBOSE:
        count = 0
        print("\n\nthe first 10 tokens in the dictionary are:")
        for k, v in dictionary.items():
            print(k, v)
            count += 1
            if count > 10:
                break

    # filter dictionary
    dictionary.filter_extremes(no_below=MIN_WORDS, no_above=MAX_PERCENTAGE, keep_n=MAX_WORDS)

    # Create Bag of Words corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if VERBOSE:
        this_row = bow_corpus[INSPECT_ROW]
        print(f'\n\nbow corpus for row #{INSPECT_ROW}: ')
        for i in range(len(this_row)):
            print(f"Word ID: {this_row[i][0]} ({dictionary[this_row[i][0]]}) appears {this_row[i][1]} time(s).")

    # are we doing a coherence test or creating a model? This is set in the DO_COHERENCE_TEST variable at the top of this file. 
    if DO_COHERENCE_TEST:
        print("\n\ncomputing coherence values for different numbers of topics, starting at", START, "and going to", LIMIT, "in steps of", STEP)
        coherence_values = compute_coherence_values(
            dictionary=dictionary,
            corpus=bow_corpus,
            texts=processed_docs,
            start=START,
            limit=LIMIT,
            step=STEP
        )

        # display graph of coherence values
        x = list(range(START, LIMIT, STEP))
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.title("Topic Coherence by Number of Topics")
        plt.show()

    else:
        # create model
        try:
            print("\n\ngoing to create model")
            lda_model = gensim.models.LdaMulticore(
                bow_corpus,
                num_topics=NUM_TOPICS,
                id2word=dictionary,
                passes=PASSES,
                workers=WORKERS,
            )
            #print the topics found by the model
            print("model created, here are the topics")
            for idx, topic in lda_model.print_topics(-1):
                print('Topic: {} \nWords: {}'.format(idx, topic))

        except Exception as e:
            print("YIKES something went wrong, quitting")
            print(str(e))
            quit()



    # save the model to disk
    if not DO_COHERENCE_TEST:
        lda_model.save('lda_model.gensim')
        dictionary.save('lda_model.dict')
        print("Model and dictionary saved to disk.")
        print("Model saved to disk.")

#With the model and dict saved, use Assign_topic_scores.py to assign topic scores 