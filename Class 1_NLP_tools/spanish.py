import re
import unicodedata
import nltk
from nltk.stem.snowball import SnowballStemmer
import spacy
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import en_core_web_sm
from tqdm import tqdm
from vtext.tokenize_sentence import UnicodeSentenceTokenizer

def eliminar_puntuacion(string): 
  string = re.sub('[´.:(),;"\']', '', string)
  return string

def eliminar_num(string): 
  string = re.sub('\d+', 'numero', string)
  return string

def uncased(text):
  return text.lower()

def eliminar_acentos(text):
  return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def eliminar_espacios(string):
  string = re.sub('  +', ' ', string)
  string = re.sub('\n', '', string)
  return string

def eliminar_especiales(string):
  string = re.sub('[@%\¡\!]', '', string)
  string = re.sub('[\$.]', '', string)
  return string
def stem(text):
  stemmer = SnowballStemmer('spanish')
  spanish_words = [stemmer.stem(t) for t in text.split()]
  return  " ".join(spanish_words)

def lemmSpacy(text):
  nlp = spacy.load("es_core_news_sm")
  doc = nlp(text)
  return " ".join([sent.lemma_ for sent in doc]), [(entity.text, entity.label_) for entity in doc.ents]

def eliminar_stopwords(text, stopwords = stopwords.words('spanish')):
  return " ".join([word for word in text.split() if word not in stopwords])

def eliminar_mails_usuarios(text):
  #removing mails
  text = re.sub('\S+@\S+','',text)
  #removing twitter user
  text = re.sub('@[^\s]+','',text)
  return text

def wordTokenize(text):
  return word_tokenize(text)

def sentenceTokenize(text, modo = nltk):
  if modo== "nltk":
    return nltk.sent_tokenize(text)
  tokenizer = UnicodeSentenceTokenizer()
  return tokenizer.tokenize(text)

#########################################################
#          ADAPTAR LA FUNCIÓN A CONTINUACIÓN            #
#########################################################
def text_preprocessing(text, puntuacion = True, mails_usuarios= True, especiales=True, espacios_extra = True, normalize = True, 
  accent = True, stemming = False, lemmatization = 'spacy', stopwords = True, 
  custom_stopwords = None, modo = 'wordtoken'):

  ###### ELIMINACIÓN ###########
  if puntuacion is True:
    text = eliminar_puntuacion(text)
  if mails_usuarios is True:
    text = eliminar_mails_usuarios(text)
  if especiales is True:
    text =eliminar_especiales(text)
  if espacios_extra is True:
    text = eliminar_espacios(text)

  ###### STOPWORDS ###########
  if stopwords is True:
    text = eliminar_stopwords(text)
  if custom_stopwords is not None:
    text = eliminar_stopwords(text=text, stopwords = custom_stopwords)

  ###### ACENTOS ###########
  if accent is True: 
    text = eliminar_acentos(text)

  #entity = None
  ###### STEMMING Y LEMMING ###########
  if stemming is True:
    text = stem(text)
  if lemmatization is 'spacy':
    text,entity = lemmSpacy(text)
  ###### NORMALIZAR ###########
  if normalize is True:
    text = uncased(text)
  if modo =='wordtoken':
    return wordTokenize(text),entity

  return text,entity

def corpus(dataset, puntuacion = True, mails_usuarios= True, especiales=True, espacios_extra = True, normalize = True, 
  accent = True, stemming = False, lemmatization = 'spacy', stopwords = True, 
  custom_stopwords = None, modo = 'wordtoken'):
  text=[]
  entity=[]
  for j in tqdm(range(len(dataset))):
    a,b = text_preprocessing(dataset[j], puntuacion, mails_usuarios, especiales, espacios_extra, normalize, accent, stemming, lemmatization, stopwords, custom_stopwords, modo)
    text.append(a)
    entity.append(b)
  text = list(filter(lambda x: x != [], text))
  entity = list(filter(lambda x: x != [], entity))
  return text,entity