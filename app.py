from flask import Flask, request,render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# loading the file
cv= pickle.load(open('count_vector.pkl','rb'))
feature_names =  pickle.load(open('feature_names.pkl','rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pkl','rb'))


stops_words = set(stopwords.words('english'))
new_words = [
    'fig', 'figure', 'image', 'sample', 'using', 'show', 'result',
    'two', 'three', 'four', 'five', 'seven', 'eight', 'nine'
]
stops_words = list(stops_words.union(new_words))

#custom function
def preprocess(txt):
    txt = txt.lower()  # tolowercase
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(r'[^a-zA-Z ]', ' ', txt)
    txt = re.sub(r'\b(html|page|bold|parent|false)\b', ' ', txt)
    txt = nltk.word_tokenize(txt)  # tokenize
    txt = [word for word in txt if word not in stops_words]  # remove stopwords
    txt = [word for word in txt if len(word) > 3]
    lemmatizer = WordNetLemmatizer()
    txt = [lemmatizer.lemmatize(word) for word in txt]

    return ' '.join(txt)


def get_keywords(docs, topN=10):
    # getting word count and importance
    docs_words_count = tfidf_transformer.transform(cv.transform([docs]))

    # sorting sparse matrix
    docs_words_count = docs_words_count.tocoo()
    tuples = zip(docs_words_count.col, docs_words_count.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    # getting top 10 keyworsds
    sorted_items = sorted_items[:topN]

    score_vals = []
    features_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        features_vals.append(feature_names[idx])

    # final result
    results = {}
    for idx in range(len(features_vals)):
        results[features_vals[idx]] = score_vals[idx]
    return results

# routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract_keywords',methods=['POST'])
def extract_keywords():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html',error='no files selected')
    if file:
        file = file.read().decode('utf-8',errors='ignore')
        cleaned_file = preprocess(file)
        keywords = get_keywords(cleaned_file,20)
        return render_template("keywords.html",keywords=keywords)
    return render_template('index.html')


@app.route('/search_keywords',methods=['POST'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
       keywords = []
       for keyword in feature_names:
           if search_query.lower() in keyword.lower():
              keywords.append(keyword)
              if len(keywords) == 20:
                 break
       print(keywords)
       return render_template('keywordslist.html',keywords=keywords)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
