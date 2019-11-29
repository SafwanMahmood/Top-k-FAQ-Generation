from nltk.corpus import stopwords
from dateutil import parser
from nltk.stem.porter import PorterStemmer
import re
from nltk.tokenize import WordPunctTokenizer

DAYS_PATTERN = '(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thur|thurs|fri|sat|sun)'
MONTHS_PATTERN = '(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
DELIMITERS_PATTERN = '([/\:\-\,\s\_\+\@]+)'
DELIMITERS_PATTERN_INTEGER_DATE = '([/\:\-])'
DIGITS_MODIFIER_PATTERN = '([0-9]+st|[0-9]+th|[0-9]+rd|[0-9]+|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth|tenth)'
DIGITS_PATTERN = '[0-9]+'


class Preprocess:

    def __init__(self, questionList):
        self.questionList = questionList
        self.questionTokens = []
        return 

    def preprocess_data(self):
        # question
        for question in self.questionList:
            urls, email, date, question = self.regexs(question)
            question_tokens = WordPunctTokenizer().tokenize(question)
            question_tokens = self.stopwords_remover(question_tokens)
            question_tokens = self.stem(question_tokens)
            question_tokens += list(urls)
            question_tokens += list(email)
            question_tokens += list(date)
            self.questionTokens.append(question_tokens)

        return self.questionTokens

    '''
        removes stopwords from the tokens created
        and also checks if req length of words is greater than
        2.
    '''

    def stopwords_remover(self,corpus):
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in corpus if w not in stop_words and len(w) >= 2]
        #second filter
        filtered_sentence = [w for w in filtered_sentence if re.match('[a-z0-9_]+', w)]
        return filtered_sentence

    '''
        This function finds urls and corpus and email and removes them from the corpus
    '''

    def regexs(self, corpus):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', corpus)
        corpus = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', corpus)
        email = re.findall('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*[.][a-zA-Z]+', corpus)
        corpus = re.sub('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*[.][a-zA-Z]+', '', corpus)
        date, corpus = self.match_date(corpus)
        return urls, email, date, corpus

    '''
        Creates a REGEX for matching Dates
    '''

    def match_date(self, corpus):
        lister = [
            '(' + DIGITS_MODIFIER_PATTERN + DELIMITERS_PATTERN + MONTHS_PATTERN + DELIMITERS_PATTERN + DIGITS_PATTERN + ')',
            '(' + DIGITS_MODIFIER_PATTERN + DELIMITERS_PATTERN + MONTHS_PATTERN + ')',
            '(' + MONTHS_PATTERN + DELIMITERS_PATTERN + DIGITS_MODIFIER_PATTERN + DELIMITERS_PATTERN + DIGITS_PATTERN + ')',
            '(' + DIGITS_PATTERN + DELIMITERS_PATTERN_INTEGER_DATE + DIGITS_PATTERN + DELIMITERS_PATTERN_INTEGER_DATE + DIGITS_PATTERN + ')',
            '(' + MONTHS_PATTERN + DELIMITERS_PATTERN + DIGITS_MODIFIER_PATTERN + ')']
        lister_list = []
        for regex in lister:
            listery = re.findall(regex, corpus)
            for date in listery:
                try:
                    parser.parse(date[0])
                    lister_list.append(date[0])
                    corpus = re.sub(date[0], '', corpus)
                except ValueError:
                    pass
        return lister_list, corpus

    '''
        Applies stemming to the tokens using
        PorterStemmer 
    '''

    def stem(self, corpus):
        porter_stemmer = PorterStemmer()
        stemmed = []
        for i in corpus:
            stemmed.append(porter_stemmer.stem(i))
        return stemmed

