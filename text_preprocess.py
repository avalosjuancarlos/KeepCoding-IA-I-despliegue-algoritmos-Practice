import re

from num2words import num2words
import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller

from bs4 import BeautifulSoup

def digits_to_words(match):
  """
  Convert string digits to the English words. The function distinguishes between
  cardinal and ordinal.
  E.g. "2" becomes "two", while "2nd" becomes "second"

  Input: str
  Output: str
  """
  suffixes = ['st', 'nd', 'rd', 'th']
  # Making sure it's lower cased so not to rely on previous possible actions:
  string = match[0].lower()
  if string[-2:] in suffixes:
    type='ordinal'
    string = string[:-2]
  else:
    type='cardinal'

  return num2words(string, to=type)

def spelling_correction(text):
    """
    Replace misspelled words with the correct spelling.

    Input: str
    Output: str
    """
    corrector = Speller()
    spells = [corrector(word) for word in text.split()]
    return " ".join(spells)

def remove_stop_words(text):
    """
    Remove stopwords.

    Input: str
    Output: str
    """
    stopwords_set = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stopwords_set])
     
def stemming(text):
    """
    Perform stemming of each word individually.

    Input: str
    Output: str
    """
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

def lemmatizing(text):
    """
    Perform lemmatization for each word individually.

    Input: str
    Output: str
    """
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocessing(input_text):
  """
  This function represents a complete pipeline for text preprocessing.

  Input: str
  Output: str
  """

  output = str(input_text)

  # Remove TAGS HTML:
  output = BeautifulSoup(output, "html5lib").get_text()

  # Lower casing:
  output = output.lower()

  # Convert digits to words:
  # The following regex syntax looks for matching of consequtive digits tentatively followed by an ordinal suffix:
  output = re.sub(r'\d+(st)?(nd)?(rd)?(th)?', digits_to_words, output, flags=re.IGNORECASE)

  # Remove punctuations and other special characters:
  output = re.sub('[^ A-Za-z0-9]+', ' ', output)

  # Spelling corrections:
  output = spelling_correction(output)

  # Remove stop words:
  output = remove_stop_words(output)

  # Stemming:
  output = stemming(output)

  # Lemmatizing:
  output = lemmatizing(output)

  return output

