from nltk.corpus import words
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from allennlp.predictors.predictor import Predictor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import seaborn as sns
from matplotlib import rcParams
import pandas as pd
from src.utils import embeddings
from gensim.models import KeyedVectors


# Loading pre-trained models
predictor_LSTM = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
predictor_roBERTa = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz")

word_list = set(map(str.lower, words.words()))
classifier = pipeline('sentiment-analysis')


# figure size in inches
rcParams['figure.figsize'] = 15, 10
sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

a = np.array([0.1, 0.2, 0.8, 0.9])
np.round(a,0)
font_size = 25

word_embeddings = [
    "GLOVE_6B_50D",
    "GLOVE_6B_100D",
    "GLOVE_6B_200D",
    "GLOVE_6B_300D",
    "GLOVE_42B_300D",
    "GLOVE_840B_300D",
    "GLOVE_TWITTER_27B_25D",
    "GLOVE_TWITTER_27B_50D",
    "GLOVE_TWITTER_27B_100D",
    "GLOVE_TWITTER_27B_200D",
    "WORD2VEC_GOOGLE_NEWS_300D",
    "FASTTEXT_CRAWL_SUB",
    "FASTTEXT_CRAWL_VEC_300D",
    "FASTTEXT_WIKI_SUB_300D",
    "FASTTEXT_WIKI_VEC_300D",
]

def KNW(word, dictionary, n = 1000, metric = 'euclid', order = 'nearest', filter = 'nltk'):
    
    word = word.lower()
    
    ''' Here we choose the way to evaluate the distance between words
        Sorting all words by distance from input word'''
    
    if metric == 'euclid':
        dist_dict = sorted(dictionary.items(), key=lambda x: np.linalg.norm([dictionary[word] - x[1]]))
    if metric == 'manhattan':
        dist_dict = sorted(dictionary.items(), key=lambda x: np.linalg.norm([dictionary[word] - x[1]], ord = 1))
    if metric == 'cosine':
        dist_dict = sorted(dictionary.items(), key=lambda x: cosine(dictionary[word], x[1]))
    
    ''' Collecting words from sorted dict '''
    
    words = []
    for key, value in dist_dict:

    #''' It is necessary to restrict some tokens from dist_dict (with numbers or other symbols within)'''
        if filter == None:
            try:
                if key.isalpha():
                    words.append(key)
            except:
                pass

    ### Another approach is to use nltk's wordlist to filter inappropriate words from dist_dict
        if filter == 'nltk':
            if (key in word_list) and key.isalpha():
                words.append(key)
    
    ''' Here we choose what kind of words we want: nearest or farest'''
    
    if order == 'nearest':
        return words[:n]
    elif order == 'farest':
        return words[-n:]


def predict_label(word, model = 'LSTM'):
    # This function returns the probability of positive class
    # If the word was not found in dictionary returns string 'Token not found'
    result_LSTM = predictor_LSTM.predict(sentence = word)
    result_roBERTa = predictor_roBERTa.predict(sentence = word)

    if result_LSTM['token_ids'] == [1]:
        result_LSTM['probs'][0] = 'Token not found'
    if len(result_roBERTa['tokens']) > 3:
        result_roBERTa['probs'][0] = 'Token not found'
    
    if model == 'LSTM':
        return result_LSTM['probs'][0]
    if model == 'roBERTa':
        return result_roBERTa['probs'][0]
    if model == 'VADER':
        model = SentimentIntensityAnalyzer()
        score = model.polarity_scores(word)
        if score['neg'] >= score['neu'] and score['neg'] >= score['pos']:
            return 1 - score['neg']
        if score['neu'] >= score['pos'] and score['neu'] >= score['neg']:
            return 0.5
        if score['pos'] >= score['neu'] and score['pos'] >= score['neg']:
            return score['pos']
    if model == 'hugging':
        output = classifier(word)[0]
        if output['label'] == 'POSITIVE':
            return output['score']
        else:
            return 1 - output['score']


def SentimentDistribution(word, n = 1000, metric = 'cosine', order = 'nearest', filter = None, model = 'LSTM'):

    '''This function returns a dictionary. The keys are the nearest words of given one, the values
       are probabilities of positive class
       If input word does not in model's dictionary of tokens, function returns 'Input token not found'
       '''

    sentiment_of_input = predict_label(word, model = model)
    if sentiment_of_input == 'Token not found':
        return 'Input token not found'
    words = KNW(word, embeddings_index, n = n, metric = metric, order = order, filter = None)
    
    dict_of_sentiments = {}
    for elem in words:
        proba = predict_label(elem, model = model)
        if proba != 'Token not found':
            dict_of_sentiments[elem] = proba
    
    return dict_of_sentiments


def DistributionPlotting(word, n = 1000, metric = 'cosine', order = 'nearest', filter = None, model = 'LSTM'):

    '''Here we just plot the distribution of sentiments'''

    dict_of_sentiments = SentimentDistribution(word, n = 1000, metric = 'cosine', order = 'nearest', filter = None, model = 'LSTM')
    if dict_of_sentiments != 'Input token not found':
        ax = sns.displot(data=np.round(np.array(list(dict_of_sentiments.values())) * 100), kind="kde", multiple="stack")
        ax.set(xlabel="Number of observations", ylabel = "Probability of Positive class",  title='Sentiment Distribution')
    else:
        return 'Input token not found'


def PlotWordPairs(word1, word2, kind = 'hist', n = 1000, metric = 'cosine',
                  order = 'nearest', filter = None, model = 'LSTM', compute_roc_auc = False, stat_test = None, md=None):
    """Description:

    Function PlotWordPairs plots sentiment distributions of the n nearest words to the word_1 and word_2.  
    **Arguments:**  
    * word_1, word_2: Given words  
    * n: The number of nearest/farest words to analyze
    * metric: {'cosine', 'euclid', 'manhattan'} default = 'cosine'  
    Choose the metric to sort the words by distance from the given one 
    * order = {'nearest', 'farest'} default = 'nearest'  
    Choose the order of sorting words: from nearest to farest if order == 'nearest', else from farest to nearest
    * filter = {None, 'nltk'} default = None  
    Filter the embeddings from sorted list of nearest/farest words. Trys to match GloVe embeddings with some words from the given dictionary
    * model: {'LSTM', 'roBERTa', 'VADER', 'hugging'} default = 'LSTM'  
    LSTM and roBERTa models are taken from allennlp https://demo.allennlp.org/sentiment-analysis  
    VADER is from nltk sentiment analysis module  
    hugging is a default model from https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextClassificationPipeline

    * kind: {'hist', 'gaussian', 'cosine', 'tophat', 'exponential'} default = 'hist'  
    Select the kind of PairPlot. If hist - draw a histogram, else draw a density curve of chosen smooting type.

    * compute_auc: 

    * stat_test: {None, 'mannwhitneyu'} default = None
    Make a test of difference between distributions
    """

    plt.rcParams.update({'font.size': font_size})
 
    word_1_dict = SentimentDistribution(word1, n, metric, order, filter, model)
    word_2_dict = SentimentDistribution(word2, n, metric, order, filter, model)
 
    

    if word_1_dict != 'Input token not found' and word_2_dict != 'Input token not found':
        
        data_1 = list(word_1_dict.values())
        data_2 = list(word_2_dict.values())

        sorted_data_1 = np.sort(data_1)
        sorted_data_2 = np.sort(data_2)

        if compute_roc_auc == True:

            auc_1 = list(zip(data_1, np.ones(len(data_1))))
            auc_2 = list(zip(data_2, np.zeros(len(data_2))))

            total_auc_data = auc_1 + auc_2
            total_auc_data = sorted(total_auc_data, key=lambda item: item[0])

            total_auc_data = np.array(total_auc_data)

            y_score, y_true = total_auc_data[:, 0], total_auc_data[:, 1]
            
            auc = roc_auc_score(y_true, y_score)

            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)

            MW = mannwhitneyu(sorted_data_1, sorted_data_2)
            
            plt.figure(figsize = (15, 10))
            plt.plot(fpr, tpr, label=f"AUC={round(auc, 2)} , Mann Whitney p_value= {round(MW[1], 2)}")
            plt.xlabel('FPR', fontsize=font_size)
            plt.ylabel('TPR', fontsize=font_size)
            plt.title(f"AUC of {word1} vs {word2} ", fontsize=font_size)
            plt.legend(loc="upper left", prop={"size": font_size})
            plt.grid()
            plt.savefig(f"{word1}_vs_{word2}_auc{md}.pdf")



        if kind == 'hist':
            plt.figure(figsize = (15, 10))
            data = {
                word1: np.round(np.array(data_1), 2),
                word2: np.round(np.array(data_2), 2)
            } 
    
            df = pd.DataFrame(data)

            sns.histplot(data=df, x=word1, color="skyblue", label=word1, kde=True, bins=100, kde_kws={'bw_adjust' :0.05})
            sns.histplot(data=df, x=word2, color="red", label=word2, kde=True, bins=100, kde_kws={'bw_adjust' :0.05})
            
            plt.xlabel('Probability of Positive Class', fontsize=font_size)
            plt.ylabel('Number of occurances of probability', fontsize=font_size)
            plt.title(f"Sentiment Distribution of {word1} vs {word2} ", fontsize=font_size)
            plt.legend(loc="upper left", prop={"size": font_size}) 
            plt.savefig(f"{word1}_vs_{word2}_dist{md}.pdf")
            
           
        
        else:

            _, bins1, _ = plt.hist(data_1, density=1, alpha=0.5, bins="auto")
            _, bins2, _ = plt.hist(data_2, density=1, alpha=0.5, bins="auto")
            plt.close()

 
            if kind == 'gaussian':
                #
                kde_1 = KernelDensity(bandwidth=0.01, kernel='gaussian')
                kde_1.fit(np.array(data_1)[:, None])
                kde_2 = KernelDensity(bandwidth=0.01, kernel='gaussian')
                kde_2.fit(np.array(data_2)[:, None])

            if kind == 'cosine':
                #
                kde_1 = KernelDensity(bandwidth=0.01, kernel='cosine')
                kde_1.fit(np.array(data_1)[:, None])
                kde_2 = KernelDensity(bandwidth=0.01, kernel='cosine')
                kde_2.fit(np.array(data_2)[:, None])
            
            if kind == 'tophat':
                #
                kde_1 = KernelDensity(bandwidth=0.01, kernel='tophat')
                kde_1.fit(np.array(data_1)[:, None])
                kde_2 = KernelDensity(bandwidth=0.01, kernel='tophat')
                kde_2.fit(np.array(data_2)[:, None])

            if kind == 'exponential':
                #
                kde_1 = KernelDensity(bandwidth=0.01, kernel='exponential')
                kde_1.fit(np.array(data_1)[:, None])
                kde_2 = KernelDensity(bandwidth=0.01, kernel='exponential')
                kde_2.fit(np.array(data_2)[:, None])


            
            logprob_1 = kde_1.score_samples(bins1[:, None])
            logprob_2 = kde_2.score_samples(bins2[:, None])
                
            plt.figure(figsize = (15, 10))
            plt.fill_between(np.linspace(0, 1, len(bins1)), np.exp(logprob_1), alpha=0.5,
                             label = word1 + '; Total number of neighbours ' + str(len(data_1)))
            plt.fill_between(np.linspace(0, 1, len(bins2)), np.exp(logprob_2), alpha=0.5,
                             label = word2 + '; Total number of neighbours ' + str(len(data_2)))
            plt.plot(data_1, np.full_like(data_1, -0.01), '|k', markeredgewidth=5, color='green', alpha = 0.5)
            plt.plot(data_2, np.full_like(data_2, -0.01), '|k', markeredgewidth=5, color='black', alpha = 0.5)             
                
            
            plt.title('Sentiment distribution of ' + word1.upper() + ' and ' + word2.upper(), fontsize=font_size)
            plt.xlabel('Probability of Positive class', fontsize=font_size)
            
            plt.legend(loc=9, prop={"size": font_size})
            plt.grid()
            plt.savefig(f"{word1}_vs_{word2}_dist{md}.pdf")   
    else:
        return 'Input token not found'


def _load_word_embedding_model(
    file=f"../models/fasttext/crawl-300d-2M-subword.vec", word_embedding_type="fasttext"
):
    model = {}
    if file is None:
        file, *ign = embeddings.get("GLOVE_6B_300D")
    print("Loading Model")
    if word_embedding_type == "glove":
        df = pd.read_csv(file, sep=" ", quoting=3, header=None, index_col=0)
        model = {key: val.values for key, val in df.T.items()}
        print(len(model), " words loaded!")
    elif word_embedding_type == "word2vec":
        key_vec = KeyedVectors.load_word2vec_format(file, binary=True)
    elif word_embedding_type == "fasttext":
        key_vec = KeyedVectors.load_word2vec_format(file, binary=False)
    for word in key_vec.vocab:
        model[word] = key_vec[word]
    return model


for embedding_name in word_embeddings:
        em_path, em_dim, em_type = embeddings.get(embedding_name)
        embeddings_index = _load_word_embedding_model(file=em_path, word_embedding_type=em_type)
        # Number of words in dict
        
        PlotWordPairs('Angel', 'Demon', model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('Cat', 'Dog', model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('European', 'African', model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('American', 'Russian', model = 'hugging', kind = 'hist', stat_test='mannwhitneyu', compute_roc_auc=True, md=embedding_name)
        PlotWordPairs('China', 'Russia', model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('USA', 'Ghana', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('Green', 'Nuclear', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('Man', 'Woman', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('Obama', 'Trump', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('US', 'Russia', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('White', 'Black', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('Men', 'Women', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('Vegan', 'Beef', model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('German', 'Jew', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)
        PlotWordPairs('American', 'African', n =1000, model = 'hugging', kind = 'hist', compute_roc_auc=True, stat_test='mannwhitneyu', md=embedding_name)

