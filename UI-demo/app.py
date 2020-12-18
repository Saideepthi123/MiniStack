from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###################################################

total_df = pd.read_pickle("Preprocessed_questions_text.pkl")
#preprocessed_text = total_df['preprocessed_text']

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(total_df['non_stopword_removed_preprocessed_text'].values)



import flask
app = Flask(__name__)


###################################################

		
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\n", "", phrase)
    return phrase
	
	
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#print ('list of stop words:', stop_words)

def nlp_preprocessing(total_text):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        return string
		

def cleanpunc(sentence): 
    """function to clean the word of any punctuation or special characters"""
    cleaned = re.sub(r'[?|!|"|#|:|=|+|_|{|}|[|]|-|$|%|^|&|]',r'',str(sentence))
    cleaned = re.sub(r'[.|,|)|(|\|/|-|~|`|>|<|*|$|@|;|→]',r'',cleaned)
    return  cleaned
###################################################



@app.route('/')
def hello_world():
    return flask.render_template('index.html')


@app.route('/crawler')
def index():
    return flask.render_template('crawler.html')

@app.route('/predict', methods=['POST'])
def predict():

    query = request.form.to_dict()
    print(type(query ['review_text']))
    print(query ['review_text'])
    query = cleanpunc(query ['review_text'].lower())
    query = decontracted(query)
    query = nlp_preprocessing(query)
    sentence = query
    
    Query_Bow = vectorizer.transform([sentence])
    from sklearn.metrics.pairwise import cosine_similarity
    doc_dict = dict()
    for i in range(X.shape[0]):
        doc_dict[i] = cosine_similarity(X[i], Query_Bow)

    a = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True) [:10]
    ##############################################################################################
    top_items = []

    for i in range(10):
        top_items.append(a[i][0])


    ################################################################################################
    count = 1
    fw = open('SimilarPosts.txt', 'w')
    for index in top_items:
        num = str(count)
        #print (total_df.iloc[index,3])
        fw.write("\n"+"Doc_num" +num +" " + total_df.iloc[index,3] + "\n")
        count = count+1
        #print("*************************************************************************************************************")
    fw.close()

    with open('SimilarPosts.txt', 'r') as f: 
	    return flask.render_template('similar.html', text=f.read()) 


Tag_Rank = {}

def tag_crawler(url):
    source_code = requests.get(url).text
    soup = BeautifulSoup(source_code, 'html.parser')
    for tag_div in soup.find_all('div', {'class': 'post-taglist'}):
        for tag_link in tag_div.find_all('a'):
            tag = tag_link.string
            if tag in Tag_Rank:
                Tag_Rank[tag] += 1
            else:
                Tag_Rank[tag] = 1

def ques_links_crawler(base_url, end_url, page_limit):
    page_no = 1
    while page_no <= page_limit:
        page_url = base_url + str(page_no) + end_url
        source_code = requests.get(page_url).text
        soup = BeautifulSoup(source_code, 'html.parser')
        if page_no is 1:
            os.system('clear')
        print('crawling page ' + str(page_no) + ': [', end='')
        prev_len = 0
        q_no = 1
        for ques_link in soup.find_all('a', {'class': 'question-hyperlink'}):
            url = 'http://stackoverflow.com/' + ques_link.get('href')
            tag_crawler(url)
            for _ in range(prev_len):
                print('\b', end='')
            print('#', end='')
            p_cent = q_no*2
            percent = '] (' + str(p_cent) + '%)'
            prev_len = len(percent)
            print(percent, end='')
            sys.stdout.flush()
            q_no += 1
        page_no += 1


@app.route('/popular', methods=['POST'])
def pop():

    page = request.form.to_dict()
    print(type(page['pop_text']))
    print(page['pop_text'])
    page_limit = int(page['pop_text'])
    print('starting crawling...')
    ques_links_crawler('http://stackoverflow.com/questions?page=', '&sort=newest', page_limit)
    fw = open('Tags_frequency.txt', 'w')
    for key, value in sorted(Tag_Rank.items(), key=operator.itemgetter(1), reverse=True):
        try:
            fw.write(key + " : " + str(Tag_Rank[key]) + "\n")
        except TypeError:
            continue
    print('\nResult saved to file Tags_frequency.txt')
    fw.close()
    
    with open('Tags_frequency.txt', 'r') as f: 
	    return flask.render_template('populartechstack.html', text=f.read()) 

if __name__ == '__main__':
    app.run()