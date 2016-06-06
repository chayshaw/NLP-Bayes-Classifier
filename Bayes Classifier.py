import random
import time
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from textblob.classifiers import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import csv


###################################### Stem & Lem #####################################

lmtzr = WordNetLemmatizer()
lancaster = nltk.LancasterStemmer()
stemmer =  PorterStemmer()


######################################################################################

# function that removes the stop words , make the word lower case and stemm the words.
def get_list_tuples(read_file):

    stop = set(stopwords.words('english'))


    list_tuples = []
    with open(read_file,'r') as r:
        reader = csv.reader(r,delimiter=',')
        x=0
        for line in reader:
            #tabsep = line.strip().split('\r')
            #tabsep = line.strip().split('')
            msg = TextBlob(line[1])
            msg.ngrams(n=2) ######################### BI-Grams #######################

            try:
                words=msg.words
            except:
                continue
            for word in words:
                #if word not in stopwords.words() and not word.isdigit():
                if word not in stop and not word.isdigit():
                    word = word.lower()
                    #list_tuples.append((word.lower(),line[0]))

                    list_tuples.append((stemmer.stem(word),line[0]))
                    #list_tuples.append((lmtzr.lemmatize(word),line[0]))
                x+=1

        return list_tuples


###################################### Trainer  #####################################

# print the import time of the data.
print ('importing data...')
a = time.time()


################ Corpus File ###################

entire_data = get_list_tuples("CorpusFile.csv")

print ("It took " + str(time.time()-a) + " seconds to import data")
print ('data imported')

# randomize the data to do a test against the corpus
random.seed(1)
random.shuffle(entire_data)


#train = entire_data[:30]
test = entire_data[1:10] # test a random 10 sentences for classification accuracy
print ('training data')
a = time.time()


################ CLASSIFY ###################


cl = NaiveBayesClassifier(entire_data) # create the classifier cl from the Trained data


###############  Accuracy ##################
print ("It took " + str(time.time()-a) + " seconds to train data")
print ('data trained, now checking accuracy:')

accuracy = cl.accuracy(test) # Accuracy of the Classifier vs the Corpus
print ("accuracy: "+str(accuracy))


stop = set(stopwords.words('english'))
def removestopword (textwords):
    finaltext1 = []
    for word in textwords.words:
                if word not in stop and not word.isdigit():
                    word = word.lower()
                    finaltext1.append(stemmer.stem(word))
    return finaltext1

##############################################################################################



########################################## Output File #######################################

Output_file = open('OUTPUTFILE.csv', 'w')

##############################################################################################



########################################## Classify File ######################################

Output = [] # this is the array for the final output

print ("############### Final CSV Output ###############")

with open('TextToClassify.csv', 'r') as fp:
    Textreader = csv.reader(fp,delimiter=',')

    Output.append("Text" + "," + "Classification" + "," + "Accuracy")
    Output_file.write(str(Output)+ "\r\n")
    Output = []

    for line in Textreader:
        x = TextBlob(line[0]) # get the Text from Column 0

        classification = cl.classify(removestopword(x))
        probs = cl.prob_classify(removestopword(x))

        Output.append(x.raw)  # Raw Text

        Output.append(cl.classify(removestopword(x))) #classify the sentence
        Output.append(probs.prob(classification)) # Confidence level of that classification


        print (Output) # print the final output
        Output_file.write(str(Output)+ "\r\n")
        Output = []



##############################################################################################






