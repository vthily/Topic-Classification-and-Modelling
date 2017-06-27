import glob, os, string, re
from stop_words import get_stop_words
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

logging.info('Initialization')

inpath = "../corpus/TDT2_top20"
doc_for_test = '../doc_for_test/economy.txt'
outpath = "/LDA/output.txt"

stop = set(stopwords.words('english') + get_stop_words('en'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def clean(doc, stop=stop, exclude=exclude, lemma=lemma, tokenizer=tokenizer):
    doc = re.sub(r'[\n\t\d]', '', doc).strip()
    stop_free = ' '.join([i for i in tokenizer.tokenize(doc.lower()) if (i not in stop
                          and len(i)>2)])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])
    return(normalized)

texts = []
for dirr in os.listdir(inpath):
    os.chdir(inpath + '/{}'.format(dirr))
    for afile in glob.glob('*.txt'):
        adoc = open(afile)
        text = clean(adoc.read()).split()
        texts.append(text)
        adoc.close()

logging.info('Creating dictionary')

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=10, no_above=0.5)
dictionary.filter_n_most_frequent(50)
dictionary.compactify()

logging.info('Creating doc_term_matrix')
doc_term_matrix = [dictionary.doc2bow(text) for text in texts]

logging.info('Creating LDA model')
lda = models.ldamodel.LdaModel(doc_term_matrix, id2word=dictionary, num_topics=20, passes=50, iterations=5)

topics = lda.print_topics(num_words = 10)

logging.info(topics)

t_texts = open(doc_for_test).read()
cleaned_texts = clean(t_texts)
test = lda[dictionary.doc2bow(cleaned_texts.split())]

logging.info('Writing output')
output = open(outpath, "w")
output.write("Below are 20 topics generated from the corpus: \n \n")
for topic in topics:
    tpc = ' '.join(re.findall(r'"(.*?)"', topic[1]))
    output.write("Topic " + str(topic[0]) + " " + tpc + "\n")
output.write("\n" + t_texts + "\n" * 2)
output.write("\n" + cleaned_texts + "\n" *2)
for i in test:
    tpc = topics[i[0]-1][1]
    tpc = ' '.join(re.findall(r'"(.*?)"', tpc))
    output.write("Topic " + str(i[0]) + ": " + str(i[1]) + " " + tpc + "\n")
output.close()

logging.info('Finish')