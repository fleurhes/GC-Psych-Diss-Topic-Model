'''
This code loads a model, opens a CSV, and assigns topic scores to the text data.
'''

import pandas as pd
import gensim
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from custom_stop_words_pattern import my_stop_words


lda_model_filename = 'lda_model.gensim'
dictionary_filename = 'lda_model.dict'
#this can be changed to the CSV path created in the psychology_data.py file
csv_file = 'filtered_corpus.csv'
headline_column = 'abstract'

nltk.download('wordnet')
stemmer = SnowballStemmer("english")

# lemmatize and stem each word
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# clean the text before assigning topic scores
def preprocess(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in my_stop_words and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


#load the model
lda_model = gensim.models.LdaMulticore.load(lda_model_filename)
dictionary = gensim.corpora.Dictionary.load(dictionary_filename)

# open the CSV
full_df = pd.read_csv(csv_file)

# drop any with missing values in the headline_column
full_df = full_df.dropna(subset=[headline_column])

# use the full cleaned corpus
df = full_df

# assign topic scores to the text data in headline_column
def assign_topic_scores(text):
    if pd.isna(text) or not isinstance(text, str):
        return None, []

    # preprocess the text
    tokens = preprocess(text)

    # convert the tokens to the model's bag-of-words format
    bow = dictionary.doc2bow(tokens)

    # get the topic distribution for the document
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)

    # convert to a list of topic scores
    topic_scores = [0] * lda_model.num_topics
    for topic_num, score in topic_distribution:
        topic_scores[topic_num] = score

    # return the best topic and all topic scores in a single pass
    best_topic_id, best_score = max(enumerate(topic_scores), key=lambda x: x[1], default=(None, 0.0))
    return best_topic_id, topic_scores

# apply the function once per row and split the result into separate columns
topic_results = df[headline_column].apply(assign_topic_scores)
df['topic_id'] = topic_results.apply(lambda x: x[0] if isinstance(x, tuple) else None)
df['topic_scores'] = topic_results.apply(lambda x: x[1] if isinstance(x, tuple) else [])

# print the record_id, topic_id, and topic_scores for the first few rows of the dataframe to verify
print(df[['record_id', 'topic_id', 'topic_scores']].head())

# expand the full topic matrix into one column per topic
topic_columns = [f'Topic {i}' for i in range(lda_model.num_topics)]

topic_matrix = pd.DataFrame(
    df['topic_scores'].tolist(),
    columns=topic_columns,
    index=df.index
)

df = pd.concat(
    [
        df.drop(columns=['topic_scores']),
        topic_matrix
    ],
    axis=1
)

print(df[['record_id', 'year', headline_column, 'topic_id'] + topic_columns].head())

df.to_csv(
    'topic_scores.csv',
    index=False
)

print(f"Topic scores saved to topic_scores.csv")
