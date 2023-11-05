from flask import Flask, request, render_template
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag

app = Flask(__name__)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


def calculate_sentence_weight_by_position(index, total_sentences):
    """ Weight sentences by their position in the document """
    return (total_sentences - index) / total_sentences


def get_wordnet_pos(treebank_tag):
    """ Return WordNet POS tag from treebank tag """
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default case


def summarize_text(text, num_sentences=5):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    word_frequencies = FreqDist(filtered_words)

    # Incorporate TF-IDF-like weighting for words based on their frequency and inverse sentence frequency.
    tf_idf_score = {word: freq / total_sentences for word, freq in word_frequencies.items()}

    sentence_scores = {}
    for index, sentence in enumerate(sentences):
        tokenized_sentence = tokenizer.tokenize(sentence.lower())
        tagged_sentence = pos_tag(tokenized_sentence)

        for word, tag in tagged_sentence:
            if word in tf_idf_score:
                # Get WordNet POS tag
                wordnet_pos = get_wordnet_pos(tag)

                if sentence not in sentence_scores:
                    sentence_scores[sentence] = calculate_sentence_weight_by_position(index, total_sentences)

                sentence_scores[sentence] += tf_idf_score[word]

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join(summary_sentences)

    return summary


@app.route('/', methods=['GET', 'POST'])  # Changed this to match the form's action.
def index():
    summary = ""
    if request.method == 'POST':
        text = request.form['text']
        num_sentences = request.form.get('num_sentences', 5)
        num_sentences = int(num_sentences) if num_sentences else 5
        if text.strip():  # Check if text is not just whitespace
            summary = summarize_text(text, num_sentences)

    return render_template('index.html', summary=summary)



if __name__ == '__main__':
    app.run(debug=True)
