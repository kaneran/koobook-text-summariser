import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import math
import heapq

#Credit to Prateek Joshi https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/ and Usman Malik https://stackabuse.com/text-summarization-with-nltk-in-python/ for the text summarsiation solutions
#in which I combined due to wanting to yield the benefits of both approaches

#This method works by first splitting the text(passed into the method's argument) into sentences. It then checks whether this is more than 3 sentences. If not then each sentence will be appended
#to a string and this method returns this string containing the sentences. It has more than 4 sentences then it will preporcess the sentences by removing the numbers, punctations, special characters
#and making it all lower case. It then iterates through each preprocessed sentence and splits into words and each word is checked to see if it a stopword. If it is a stopword then it remove from the sentence.
#The output of that step is cleaner sentences without stopwords. For these sentences, it then goes through each word and checks whether it exists in the word frequency dictionary. If it doesn't exist then create a new record in the dictionary
#. If it does exist then it increments the frequnecy value for that particular record. This process repeats until all words have been processed. This dictionary containing the word and its frequnecy of occurence is then used to calculate
#the overall score for each sentence by iterating through each word in the sentence and check if the word exisits in the word frequnecy dictionary. If it doesn't then added nothing to the sentence score. if it does
#then get the corresponding frequency value for that word and add that to the sentence score. After processing all the words in that sentence, add the sentence along with the sentence score to the dictionary
#for sentence scores. The process repeats for all sentences. Then, gets the top 5 sentences from the sentence scores dictionary which has the highest score and these sentences are then used to create
#vector representations for each sentence. It then uses this to create a simlarity matrix. This matrix is then used to execute the PageRank algorithm which returns the page rank scores.
#This is then used to get the top ranked sentences and append it as a single string and this string is returned by this method is contains the summary of the top ranked sentences. Note that
#the page rank algorithms derives the scores based on how relevent each sentence is in contrast with other sentences.
def summarise(text, word_embeddings):
    sentences = splitIntoSentences(text)
    if len(sentences) <4:
        sentencesInOriginalForm = ""
        for s in sentences:
            sentencesInOriginalForm.append(s)
        return sentencesInOriginalForm
    else:
        sentences = preprocessSentences(sentences)
        sentences = removeStopwordsFromSentences(sentences)
        word_frequencies = findWeightedFrequencyOfOccurence(sentences)
        sentence_weighted_frequency_scores = calculateSentenceScores(word_frequencies, sentences)
        important_sentences = getSentencesBasedOnSentenceScore(sentence_weighted_frequency_scores)
        sentence_vectors = createVectorsForSentences(important_sentences, word_embeddings)
        similarity_matrix = createSimilarityMatrix(important_sentences, sentence_vectors)
        page_rank_scores = applyPageRank(similarity_matrix)
        ranked_summary = getRankedSentences(page_rank_scores, important_sentences)
        print(ranked_summary)
        return ranked_summary

#This method works by first splitting the text(passed into the method's argument) into sentences.
def splitIntoSentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

#This method works by preprocessing the sentences by removing the numbers, punctations, special characters
#and making it all lower case.
def preprocessSentences(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^A-Za-z0-9]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    return clean_sentences

#This method works by iterating through each preprocessed sentence removing the stopwords from each sentence. It this returns sentences without stopwords
def removeStopwordsFromSentences(sentences):
    clean_sentences = [removeStopwords(sentence.split()) for sentence in sentences]
    return clean_sentences

#This method works by splitting the sentence into words and each word is checked to see if it a stopword. If it is a stopword then it removes it from the sentence.
#The output of this is a cleaner sentence without stopwords.
def removeStopwords(sentence):
    stop_words = stopwords.words('english')
    new_sentence = " ".join(i for i in sentence if i not in stop_words)
    return new_sentence

#This method works by iterating through each sentence and iterating through each word and checks whether if the word is not a stopword. If it is not then it checks whether it exists in the word frequency dictionary.
#If it doesn't exist then create a new record in the dictionary. If it does exist then it increments the frequnecy value for that particular record. This process repeats until all words have been processed. At that point
#it then gets the maxmimum frequnecy value from the maximmum frequency values and derives the weighted frequecy for each word by dividing the word freeuqnecy value for that word by the maximum frequnecy value. After deriving the
# value,it then updates the value in the word frequency dictionary to be that of the weighted frequency for that word. This process repeats for all words in the word frequnecy dictionary.
#This updated dictionary containing the word and its weighted frequnecy of occurence is returned by this method.
def findWeightedFrequencyOfOccurence(clean_sentences):
    stop_words = stopwords.words('english')
    word_frequencies = {}
    sentences = ".".join(clean_sentences)
    for word in nltk.word_tokenize(sentences):
        if word not in stop_words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maxmimum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maxmimum_frequency)
    return word_frequencies


#This methods works using the dictionary ,containing the word and its weighted frequnecy of occurence, to calculate the overall score for each sentence.
#This is calculated by iterating through each word in the sentence and check if the word exists in the word frequency dictionary. If it doesn't then add nothing to the sentence score. If it does
#then get the corresponding weighted frequency value for that word and add that to the sentence score. After processing all the words in that sentence, add the sentence along with the sentence score to the dictionary
#for sentence scores. The process repeats for all sentences and this method returns the dictionary for sentence scores
def calculateSentenceScores(word_frequencies, clean_sentences):
    sentence_scores = {}
    for sentence in clean_sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
    return sentence_scores

#This method works by getting the top 5 sentences from the sentence scores dictionary which has the highest score and this is what is returned by this method
def getSentencesBasedOnSentenceScore(sentence_scores):
    sentences = heapq.nlargest(5, sentence_scores, key = sentence_scores.get)
    return sentences

#This method works by reading the text file containg the words followed by its vector representation on the same row. It then iterates through reach line in the text file and
#splits values such it can access the word and vector value separately.It uses the vector value to ge the coefficient of the vector value and this along with the word is added to the word embeddings dictionary.
#After reading all lines from the file, this method returns this dictionary.
def extractWordEmbeddings():
    word_embeddings = {}
    f = open('Word embeddings/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

#This method works by creating vector representations for each sentence. This is done by iterating through each sentence and checks whether a sentence contains words. If it does then it iterates through each word in that sentence.
#For each word, it gets the vector value from the word embeddings dictionary. This process repeats for each word in that sentence and all the derived values are summed and divided by the number of words in the sentence
#in which the word belongs to plus 0.001 and this is the vector for that sentence. However, if the sentence does not contain any words then the vector representation for that sentence will be a default value.
#After deriving the sentence vector value for that sentence, it then adds this to list of sentence vectors. After iterating through each sentence and added its sentence vector to the list, this list
#is returned by this method.
def createVectorsForSentences(sentences, word_embeddings):
    sentence_vectors = []
    for sentence in sentences:
        if len(sentence) != 0:
            vector = sum([word_embeddings.get(word, np.zeros((100,))) for word in sentence.split()])/(len(sentence.split())+0.001)
        else:
            vector = np.zeros((100,))
        sentence_vectors.append(vector)
    return sentence_vectors

#This method works by using the sentences along with the list of sentence vectors to create a simlarity matrix. This is done by using the number of sentences to intiialise the simarlity matrix.
#It then populates the similarity matrix by iterating through each sentence and using the index of the sentence get the sentence vector representation and this is used to compute the conseine simiarlity score
#and this is added to the simiarlity matrix. After iterating through each sentence, this method returns the similarity matrix.
def createSimilarityMatrix(sentences, sentence_vectors):
    similarity_matrix = np.zeros([len(sentences), len(sentences)])

    #Initialise matrix with cosine similarity scores
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    return similarity_matrix

#This method works by using the similarity matrix to execute the PageRank algorithm which returns the page rank scores and this what the method returns.
def applyPageRank(similarity_matrix):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    return scores

#This method works by getting the top ranked sentences and appending it as a single string and this string is returned by this method which is contains the summary of the top ranked sentences.
#This is done by deriving the limit which represents the maximum number of sentences to get. It then ranked the sentences based on the page rank scores . It then iterates through each
#ranked sentence ,that's within the ranhe of the previous derived limit, and adds it to the list of sentences. After adding the ranked sentences to the list, it then iterates through of sentence in that list
#and splits the sentence by the first letter whilst still storing the remaining letter in the sentence. This is done so the first letter can be accessed and made to upper case and this
#is appended to the remaining letters of that sentence to produce the formatted sentence. This formated sentence is then added to the string called ranked_summary. This process repeats for each top ranked sentence and all the formatted sentences
#being added to the ranked_summary string are seperated by using ". ". After iterating through all the top ranked sentences, the method returns the ranked summary string and this is what is sent
#back to the android application via TCP connection.
def getRankedSentences(scores, sentences):
    top_limit = math.floor(len(sentences) - (len(sentences)/3))
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    top_sentences = []
    top_sentence = []

    for i in range(top_limit):
        top_sentences.append(ranked_sentences[i][1])

    for sentence in top_sentences:
        first_letter = sentence[0]
        remaining_chars = sentence[1: (len(sentence))]
        top_sentence.append(first_letter.upper() + remaining_chars)
    ranked_summary = ". ".join(top_sentence)
    return ranked_summary