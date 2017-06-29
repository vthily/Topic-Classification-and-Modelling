import glob, re, string, logging
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from stop_words import get_stop_words

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

#Loading data

logging.info("Initialization")
inpath='../corpus/TDT2_top20'
outpath='/nn/output.txt'

stop = set(stopwords.words('english') + get_stop_words('en'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def clean(doc, stop=stop, exclude=exclude, lemma=lemma, tokenizer=tokenizer):
    doc = re.sub(r'[\n\t\d]', '', doc).strip()
    stop_free = ' '.join([i for i in tokenizer.tokenize(doc.lower()) if (i not in stop
                          and len(i)>3)])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])
    return(normalized)

# Load target label
name_list=[]
namefile=open('../corpus/name_list.txt', 'r')
for line in namefile:
	name_list.append(line.rstrip())

all_texts=load_files(inpath)
logging.info("Finish Initialization")

data_train, data_test, target_train, target_test = train_test_split(all_texts.data, all_texts.target,
                                                                    test_size=0.2)

logging.info("training "+str(len(data_train)))
logging.info("test "+str(len(data_test)))


model = MLPClassifier(random_state=0, verbose=True, hidden_layer_sizes=(20, 20, ), activation='relu',
                      max_iter=1, warm_start=1)

text_clf = Pipeline(
        [('vect', CountVectorizer(stop_words=stop, preprocessor=clean)), 
         ('tfidf', TfidfTransformer()), 
         ('clf', model),
         ])


# Change number of iteration to change number of epochs for training
test_loss = []
training_loss = []
for i in range(20):
    text_clf.fit(data_train, target_train)
    training_loss.append(metrics.log_loss(target_train, text_clf.predict_proba(data_train)))
    print('Training Loss:', training_loss[-1])
    test_loss.append(metrics.log_loss(target_test, text_clf.predict_proba(data_test)))
    print('Test_loss:', test_loss[-1])

plt.figure()
plt.plot(np.arange(len(training_loss)), training_loss, 'r-', label='Training Loss')
plt.plot(np.arange(len(test_loss)), test_loss, 'b-', label='Test Loss')
plt.ylim((0, 3))
plt.xlabel('Number of iteration')
plt.ylabel('Cross entropy Loss')
plt.legend()

predicted = text_clf.predict(data_test)

logging.info('Accuracy:', metrics.accuracy_score(target_test, predicted))
logging.info('Cross-Entropy:', metrics.log_loss(target_test, text_clf.predict_proba(data_test)))
print(metrics.classification_report(target_test, predicted, target_names = all_texts.target_names))


logging.info('Writing output')
#predict new doc
outfile = open('/nn/output.txt', 'w+')

for afile in glob.glob("../doc_for_test/*.txt"):
    
    newfile = open(afile,'r')
    newdoc = newfile.read().replace('\n', ' ')
    newdoc = newdoc.replace('\t', ' ')
    outfile.write('\n')
	
    outfile.write(str(afile)+'\n\n')
    outfile.write(newdoc+'\n\n')

    result=text_clf.predict_proba([newdoc])[0]
    
    result_metrics=[]
    
    for i in range(len(result)):
        result_vector=[]
        result_vector.append(name_list[i])
        result_vector.append(result[i])
        result_metrics.append(result_vector)
        
    result_metrics_sorted = sorted(result_metrics, key=lambda a_entry: a_entry[1], reverse=True) 

    for a_vector in result_metrics_sorted:
        outfile.write(str(a_vector[0])+'\t')	
        a_vector[1]=round(a_vector[1],5)
        outfile.write(" {}".format(a_vector[1]))
        outfile.write('\n')
        
    newfile.close()
outfile.close()

logging.info('Finish')