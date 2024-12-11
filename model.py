from nltk import ngrams
from collections import defaultdict,Counter
from indicnlp.tokenize import indic_tokenize
import re
import math

# Load and preprocess Malayalam text from a text file

def preprocess_input(sentence):
    corpus = re.sub(r'[^\u0D00-\u0D7F\s]', '', sentence).lower()  # Keep Malayalam chars only
    tokens = list(indic_tokenize.trivial_tokenize(corpus, lang='ml'))
    return tokens

def load_data_from_files(file_range, directory):
    """
    Loads and processes text data from files in the given range (e.g., 1.txt to 24.txt).
    
    Args:
    - file_range (tuple): The range of files to read (e.g., (1, 24)).
    - directory (str): The directory where the text files are stored.
    
    Returns:
    - tokens (list): A list of tokenized words from all files.
    """
    corpus = ""
    
    for i in range(file_range[0], file_range[1] + 1):
        filename = f"{directory}/{i}.txt"  # Construct the file path
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            for line in lines:
                line = line.strip()  # Remove leading/trailing whitespaces
                if line:  # Ignore empty lines
                    corpus += line + " "
        except FileNotFoundError:
            print(f"File {filename} not found!")
    
    # Preprocessing: Remove non-Malayalam characters and tokenize
    
    
    return preprocess_input(corpus)

# Example usage: read files from 1.txt to 24.txt in the 'Dataset' directory


tokens_ml = load_data_from_files((1, 24), 'Dataset')


# Create bigrams and trigrams
bigrams_ml = list(ngrams(tokens_ml, 2))
trigrams_ml = list(ngrams(tokens_ml, 3))
unigrams_ml = list(ngrams(tokens_ml, 1))
# Frequency counts for bigrams and trigrams
unigram_freq_ml = Counter(unigrams_ml)
bigram_freq_ml = Counter(bigrams_ml)
trigram_freq_ml = Counter(trigrams_ml)
word_freq_ml = Counter(tokens_ml)

# Continuation Probability (count of unique preceding bigram heads)
def continuation_prob_ml(word):
    unique_preceding = set(w1 for (w1, w2) in bigram_freq_ml if w2 == word)
    return len(unique_preceding) / len(bigram_freq_ml)

# Kneser-Ney Smoothing for Trigrams
def kneser_ney_prob_ml(w1, w2, w3, d=0.75):
    trigram_count = trigram_freq_ml[(w1, w2, w3)]
    bigram_count = bigram_freq_ml[(w1, w2)]

    if bigram_count > 0:
        p_trigram = max(trigram_count - d, 0) / bigram_count
    else:
        p_trigram = 0

    lambda_factor = d * bigram_count / (bigram_count + unigram_freq_ml[w2])
    p_continuation = continuation_prob_ml(w3)

    p_kneser_ney = p_trigram + lambda_factor * p_continuation
    return p_kneser_ney

# Prediction function using bigrams or trigrams (higher-order n-grams)
def predict_next_word_ngram(words, top_n=5):
    candidates = []

    # print("top: ", words)
    if len(words) == 1:
        # print("Bigram model (single word as context)")
        word = words[0]
    # Collect candidate words with probabilities
        candidates = [
            (w3, kneser_ney_prob_ml(w1, w2, w3))
            for (w1_t, w2_t, w3) in trigram_freq_ml
            if w1_t == w1 and w2_t == w2
        ]
        # print(candidates)
    elif len(words) >= 2:
        # Trigram model (two words as context)
        w1, w2 = words[-2], words[-1]

    # Collect candidate words with probabilities
        # candidates = [
        #     (w3, kneser_ney_prob_ml(w1, w2, w3))
        #     for (w1_t, w2_t, w3) in trigram_freq_ml
        #     if w1_t == w1 and w2_t == w2
        # ]
        print("c1: ", candidates)
        word1, word2 = words[-2], words[-1]
        candidates = [
            (w3, trigram_freq_ml[(word1, word2, w3)] / bigram_freq_ml[(word1, word2)])
            for (w1, w2, w3) in trigram_freq_ml if w1 == word1 and w2 == word2
        ]
        if (len(candidates) == 0) and (isinstance(words,list)) :
            # print(type(words))
            # print(words)
            words.pop(0)
            return predict_next_word_ngram(words)
    else:
        return {"error": "Insufficient words for prediction"}
    
    # Sort candidates by probability
    # print("c1: ", candidates)
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates[:top_n]

# Calculate Perplexity
def calculate_perplexity_ml(test_sentences):
    total_log_prob = 0
    total_words = 0

    for sentence in test_sentences:
        words = preprocess_input(sentence)  # Preprocess as in training
        n = len(words)
        
        # Skip sentences too short for trigram evaluation
        if n < 3:
            continue
        
        for i in range(2, n):
            w1, w2, w3 = words[i - 2], words[i - 1], words[i]
            prob = kneser_ney_prob_ml(w1, w2, w3)
            
            # Avoid log(0) by using a small probability value
            if prob > 0:
                total_log_prob += math.log(prob)
            else:
                total_log_prob += math.log(1e-10)
            
            total_words += 1
    
    if total_words == 0:
        return float('inf')  # Avoid division by zero
    
    # Calculate perplexity
    perplexity = math.exp(-total_log_prob / total_words)
    return perplexity


def predict(words):
    word_list = words.split()
    return predict_next_word_ngram(word_list)

