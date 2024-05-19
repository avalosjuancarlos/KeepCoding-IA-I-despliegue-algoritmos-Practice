from sklearn.feature_extraction.text import CountVectorizer

def extract_BoW_features(words_train, words_test, vocabulary_size=5000):
  """Extract Bag-of-Words for a given set of documents"""

  vectorizer = CountVectorizer(max_features=vocabulary_size,
          preprocessor=lambda x: x, tokenizer=lambda x: x)  # already preprocessed
  features_train = vectorizer.fit_transform(words_train).toarray()

  features_test = vectorizer.transform(words_test).toarray()

  vocabulary = vectorizer.vocabulary_

  return features_train, features_test, vocabulary
