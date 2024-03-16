"""
building k-means clustering on wiki data.
2 March 2024
"""

### (0) initial libraries to import and why: ###

# (a) to access file paths
import os

# (b) to manipulate json data
import json 

# (c) import vectorizers for cleaned data
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader

# (d) to use KMeans, MiniBatchKMeans algorithms on preprocessed data
from sklearn.cluster import KMeans, MiniBatchKMeans

# (e) to measure quality of cluster assignment
from sklearn.metrics import silhouette_score

# (f) for mathematical efficiency
import numpy as np 

# (g) to save cluster data in useful file format
import pickle

# (h) to save the kmeans model for later inference
import joblib



### (1) load in data ###
#  -  sort from given structure of json {"filename":"mbid.txt", "mid":"b'mid", "paragraphs": ["str",]}
#  -  to desired structure of dictionary of midi cluster assignments {cluster : [mbid, "whole-wikidoc string"]} in (2)

# (a) helper function to convert .txt to .json data
def txt_to_json(input_file):
    print(input_file)

    # extract the filename and extension
    filename, ext = os.path.splitext(input_file)
    print(filename, ext)

    # check if the file extension is .txt
    if ext == '.txt':
        print("YES")

        # rename the file with .json extension
        new_filename = filename + '.json'
        os.rename(input_file, new_filename)
        return new_filename   
    else:
        print("calling to (1)(a) unneccessarily with json file")
        return input_file
    

# (b) actually load and preprocess data
def load_data(data_path):
    """
    load and preprocess data from given data_path file 

    params: 
    data_path (str): path to directory containing data files

    returns:
    wiki_data_preproc (dict): {mbid: "whole-wikidoc string"}
    """
    
    # initialize return dictionary
    wiki_data_preproc = {}

    # loop over every file in data directory
    for filename in os.listdir(data_path):
       
        # ensure that file is json, convert .txt to .json
        if filename.endswith(".txt"):
            print("converting .txt file to .json...")
            filename = txt_to_json(filename)
            
        # open and read
        with open(os.path.join(data_path, filename), 'r') as data:

            # get json data
            json_data = json.load(data)
            mbid = json_data["filename"][:-4]  # shave off the '.txt' last 4 chars
            text = " ".join(json_data["paragraphs"])

            # only save if paragraphs are non-trivial
            if len(text) > 10:
                wiki_data_preproc[mbid] = text

    # return the object
    return wiki_data_preproc



### (2) k-means algorithm with scikit-learn, using TF-IDF vectorizer ###

def run_kmeans(data_preproc):
    """
    find kmeans clusters with varying values for K. saves model, vectorizer for later inference.

    params:
    data_preproc (dict)

    returns:
    clusters (dict), best_k (int)
    """

    # convert data to numerical vectors with TF-IDF vectorizer
    filler_words = ["the", "and", "to", "is", "in", "it", "of", "that", "this"]  # add more as needed
    vectorizer = TfidfVectorizer(stop_words=filler_words)
    vectorized_data = vectorizer.fit_transform(data_preproc.values())
    print("data has been vectorized")

    # determine optimal k-value by silhouette scores of k-value options; initialize list to track
    silhouette_scores = []

    # set a range for k values: 8, 16, 32, 64, 128
    k_values = [8, 16, 32, 64, 128, 256, 512, 1048]

    # loop over values and try
    for k in k_values:
        print(f"trying {k} clusters...")
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vectorized_data)
        labels = kmeans.labels_

        # get silhouette score
        s_score = silhouette_score(vectorized_data, labels)
        silhouette_scores.append(s_score)
        print(f"silhouette_score for {k} clusters: {s_score}")

    # choose the cluster size with the highest silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]

    # run kmeans on this num clusters
    best_kmeans = KMeans(n_clusters=best_k)
    best_kmeans.fit(vectorized_data)
    best_labels = kmeans.labels_
    best_s_score = silhouette_score(vectorized_data, best_labels)
    print(f"s_score for {best_k} clusters: {best_s_score}")

    # save kmeans model and vectorizer for later inference –– uncomment as necessary
    # joblib.dump(best_kmeans, "best_kmeans_model.joblib")
    # joblib.dump(vectorizer, "vectorizer.joblib")

    # build dictionary of clusters to return
    print("building clusters")
    clusters = {}
    for mbid, label in zip(data_preproc.keys(), labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((mbid, data_preproc[mbid]))

    print(f"returning clusters, best_k={best_k}")
    return clusters, best_k



### (3) save clusters helper function ###
#  -  in useful fashion to pass into AMT transformer architecture 
#  -  aim to prepend k-means cluster as initial token label   

def save_clusters(clusters, filename):
    """
    Save clusters and corresponding documents to a pickle file.

    Parameters:
    clusters (dict): Dictionary containing clusters {cluster : [(mbid, "whole-wikidoc string")]}.
    filename (str): Name of the pickle file to save.
    """
    with open(filename, 'wb') as f:
        pickle.dump(clusters, f)



### (4) k-means with Word2Vec ###

# (a) need some other libraries –– uncomment as necessary
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# from string import punctuation

# (b) need a function to vectorize before kmeans
def vectorize(data_preproc):
    """
    use the gensim glove-wiki-gigiaword w2v model to vectorize data to then pass into kmeans.
    lowercase, clean and remove stop words

    params: 
    data_preproc (dict): {mbid: "whole-wikidoc string"}

    returns:
    vectorized_data: np.array(doc_vecs)
    mbids: np.array(mbids)
    """

    # first, load the model
    w2v_model = gensim.downloader.load('glove-wiki-gigaword-50')

    # define the stopwords from nltk
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # stop_words = set(stopwords.words('english'))

    # remove punctuation
    stop_words.update(set(punctuation))

    # initialize list to store document vectors, mbids
    doc_vecs = []
    mbids = []

    # iterate over the preprocessed wiki data dict {mbid: "whole doc string"}
    print("iterating, tokenizing and vectorizing")
    for mbid, doc in data_preproc.items():
        
        # tokenize the document
        tokens = word_tokenize(doc.lower())

        # remove stop words
        tokens = [tok for tok in tokens if tok not in stop_words]

        # intialize list for this iteration of word vectors
        word_vecs = []

        # vectorize with w2v 
        for tok in tokens:
            if tok in w2v_model:
                word_vecs.append(w2v_model[tok])

        # check if we actually got word vecs
        if not word_vecs:
            print("error: no word vectors in this iteration...")
            continue

        # average the vectors to get a single document vector
        doc_vector = np.mean(word_vecs, axis=0)

        # add this document vector and mbid to the outer list
        doc_vecs.append(doc_vector)
        mbids.append(mbid)

    # convert the lists to numpy arrays
    vectorized_data = np.array(doc_vecs)
    mbids = np.array(mbids)

    # save the vectorized data and corresponding MBIDs to a file
    np.savez('w2v_vectorized_data', vectorized_data=vectorized_data, mbids=mbids)

    print("data vectorized!")
    return vectorized_data, mbids


# (c) now we have vectorized data, so run kmeans
def kmeans_w2v(vectorized_data, mbids, K):
    """
    the TFIDF vectorizer is insufficient; need a richer vectorizer 

    """

    # set and run kmeans model
    kmeans = KMeans(n_clusters=K)
    print(f"running kmeans on {K} clusters...")
    kmeans.fit(vectorized_data)

    # get cluster assignments for mbid 
    cluster_assignments = kmeans.labels_
    mbid_clusters = {mbid: cluster for mbid, cluster in zip(mbids, cluster_assignments)}

    # get silhouette scores
    silhouette = silhouette_score(vectorized_data, cluster_assignments)
    print(f"silhouette score for {K} clusters: {silhouette}")

    # # save model and clusters
    # joblib.dump(kmeans, 'w2v_kmeans_model')
    # save_clusters(mbid_clusters, "mbid_w2v_clusters")

    return mbid_clusters, silhouette



### (5) GPT-2 NN approach to feed into kmeans ###
#  -  empirically this seems to be the best approach
#  -  also the most computationally expensive

# (a) additional libraries to import
from transformers import GPT2Tokenizer, GPT2Model

# (b) load data from saved when possible 
def load_GPT_vectorized_data(file_path):
    """
    Load the saved BERT vectorized data.

    Params:
    file_path (str): Path to the saved npz file containing vectorized data.

    Returns:
    embeddings: numpy array of BERT embeddings.
    mbids: numpy array of MBIDs.
    """

    print(f"loading GPT vectorized data from {file_path}")
    # Load the npz file
    loaded_data = np.load(file_path)

    # Extract embeddings and mbids
    embeddings = loaded_data['vectorized_data']
    mbids = loaded_data['mbids']

    return embeddings, mbids


# (c) need a function to tokenize and vectorize the data
def GPT_vectorize(data_preproc, save_file=None):
    """
    clean, lowercase, tokenize etc. the initially preprocessed data

    params:
    data_preproc (dict): {mbid: "whole-wikidoc string"}

    returns:
    embeddings to use in kmeans
    """

    # if we have already run it, just open the file
    if save_file:
        embeddings, mbids = load_GPT_vectorized_data(save_file)
        print("loaded GPT vectorized data!")

    # otherwise, run the BERT approach
    else:

        # load pretrained BERT tokenizer and model
        print("loading GPT tokenizer and model")
        GPT_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding=True)
        GPT_tokenizer.pad_token = '[PAD]'
        GPT_model = GPT2Model.from_pretrained('gpt2')
        print("loaded!")

        # initialize lists to store embeddings, mbids
        embeddings = []
        mbids = []

        # iterate over preprocessed data and get embeddings
        print("iterating...")
        iter_count = 0
        for mbid, doc in data_preproc.items():

            # just to keep track of where we are
            iter_count += 1
            if iter_count % 1000 == 0:
                print(f"iteration number {iter_count}")
                print(f"    currently working on {mbid} : {doc[:20]}...")

            # tokenize the doc text, get pytorch tensors
            encoded_input = GPT_tokenizer(doc, return_tensors="pt", padding=True, truncation=True)

            # generate the embeddings
            with torch.no_grad():

                # get outputs from GPT model
                outputs = GPT_model(**encoded_input)

                # get the [CLS] (classification) token for sequence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            # append the embedding and corresponding mbid to the lists
            embeddings.append(cls_embedding)
            mbids.append(mbid)

        # convert the lists of embeddings and mbids to numpy arrays
        embeddings = np.array(embeddings)
        mbids = np.array(mbids)

        # save the vectorized data and corresponding MBIDs to a file
        np.savez('GPT_vectorized_data.npz', vectorized_data=embeddings, mbids=mbids)

    # compress the embeddings so that kmeans will work
    embeddings = np.concatenate(embeddings, axis=0)

    print("data GPT embedded!")
    return embeddings, mbids


# (d) run kmeans with bert
def kmeans_GPT(embeddings, mbids, K, save=False):

    # set and run kmeans model
    kmeans = KMeans(n_clusters=K)
    print(f"running kmeans on {K} clusters...")
    kmeans.fit(embeddings)

    # get cluster assignments for mbid 
    cluster_assignments = kmeans.labels_
    mbid_clusters = {mbid: cluster for mbid, cluster in zip(mbids, cluster_assignments)}

    # get silhouette scores
    silhouette = silhouette_score(embeddings, cluster_assignments)
    print(f"silhouette score for {K} clusters: {silhouette}")

    # mean and sd ?

    # save model and clusters
    if save:
        joblib.dump(kmeans, f'GPT_{K}k_means_model.joblib')
        save_clusters(mbid_clusters, f"mbid_GPT_{K}_clusters.pickle")

    return mbid_clusters, silhouette
    


### (6) BERT NN approach to feed into kmeans ###

# (a) additional libraries to import
from transformers import BertTokenizer, BertModel
import torch

# (b) load data from saved when possible 
def load_BERT_vectorized_data(file_path):
    """
    Load the saved BERT vectorized data.

    Params:
    file_path (str): Path to the saved npz file containing vectorized data.

    Returns:
    embeddings: numpy array of BERT embeddings.
    mbids: numpy array of MBIDs.
    """

    print(f"loading BERT vectorized data from {file_path}")
    # Load the npz file
    loaded_data = np.load(file_path)

    # Extract embeddings and mbids
    embeddings = loaded_data['vectorized_data']
    mbids = loaded_data['mbids']

    return embeddings, mbids


# (c) need a function to tokenize and vectorize the data
def BERT_vectorize(data_preproc, save_file=None):
    """
    clean, lowercase, tokenize etc. the initially preprocessed data

    params:
    data_preproc (dict): {mbid: "whole-wikidoc string"}

    returns:
    embeddings to use in kmeans
    """

    # if we have already run it, just open the file
    if save_file:
        embeddings, mbids = load_BERT_vectorized_data(save_file)
        print("Loaded BERT vectorized data!")

    # otherwise, run the BERT approach
    else:

        # load pretrained BERT tokenizer and model
        print("loading BERT tokenizer and model")
        BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_model = BertModel.from_pretrained('bert-base-uncased')
        print("loaded!")

        # initialize lists to store embeddings, mbids
        embeddings = []
        mbids = []

        # iterate over preprocessed data and get embeddings
        print("iterating...")
        for mbid, doc in data_preproc.items():

            # tokenize the doc text, get pytorch tensors
            encoded_input = BERT_tokenizer(doc, padding=True, truncation=True, return_tensors='pt')

            # generate the embeddings
            with torch.no_grad():

                # get outputs from BERT model
                outputs = BERT_model(**encoded_input)

                # get the [CLS] (classification) token for sequence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            # append the embedding and corresponding mbid to the lists
            embeddings.append(cls_embedding)
            mbids.append(mbid)

        # convert the lists of embeddings and mbids to numpy arrays
        embeddings = np.array(embeddings)
        mbids = np.array(mbids)

        # save the vectorized data and corresponding MBIDs to a file
        np.savez('BERT_vectorized_data', vectorized_data=embeddings, mbids=mbids)

    # compress the embeddings so that kmeans will work
    embeddings = np.concatenate(embeddings, axis=0)

    print("data BERT embedded!")
    return embeddings, mbids


# (d) run kmeans with bert
def kmeans_BERT(embeddings, mbids, K):

    # set and run kmeans model
    kmeans = KMeans(n_clusters=K)
    print(f"running kmeans on {K} clusters...")
    kmeans.fit(embeddings)

    # get cluster assignments for mbid 
    cluster_assignments = kmeans.labels_
    mbid_clusters = {mbid: cluster for mbid, cluster in zip(mbids, cluster_assignments)}

    # get silhouette scores
    silhouette = silhouette_score(embeddings, cluster_assignments)
    print(f"silhouette score for {K} clusters: {silhouette}")

    # # save model and clusters –– uncomment as necessary
    # joblib.dump(kmeans, 'BERT_kmeans_model')
    # save_clusters(mbid_clusters, "mbid_BERT_clusters")

    return mbid_clusters, silhouette
    


### (7) stack of function calls to obtain results ###

# (a) declare data path 
data_path = 'full_dataset'

# (b) load data
data = load_data(data_path)
print(len(data.keys()))

# (c) search for value of k
embeddings, mbids = GPT_vectorize(data_preproc=data, save_file='GPT_vectorized_data.npz')
print(embeddings.shape)
for k_val in [8, 16, 32, 64, 128, 256, 512]:
    clusters, silhouette = kmeans_GPT(embeddings=embeddings, mbids=mbids, K=k_val, save=False)

    # get mean and sd for k clusters
    cluster_counts = {}
    for mbid, cluster in clusters.items():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    # Convert the dictionary values to a list
    counts_list = list(cluster_counts.values())

    # Calculate mean and standard deviation
    mean_assigned = np.mean(counts_list)
    std_assigned = np.std(counts_list)

    print(f"For {k_val} clusters:")
    print(f"Mean number of MBIDs per cluster: {mean_assigned}")
    print(f"Standard deviation of number of MBIDs per cluster: {std_assigned}")
    


### other attempts and vague notes –– uncomment as neccesary ###


# # run k-means clustering
# clusters, best_k = run_kmeans(data)

# # save clusters
# save_clusters(clusters, 'clusters_data.pickle')


# try kmeans with BERT
# embeddings, mbids = BERT_vectorize(data, save_file='BERT_vectorized_data.npz')

# k = [16, 32, 64, 128, 512, 1024]
# for val in k:
#     kmeans_BERT(embeddings, mbids, val)

# kmeans_BERT(embeddings, mbids, K=5000)
# embeddings, mbids = GPT_vectorize(data_preproc=data, save_file='GPT_vectorized_data.npz')


# k = [16, 32, 64, 128, 512, 1024]
# for val in k:
#     kmeans_GPT(embeddings=embeddings, mbids=mbids, K=val)



# attempt w/ w2v approach
# vectorized_data, mbids = vectorize(data)
# vectorized_data, mbids = "fill this in with the way to get the data from w2v_vectorized_data.npz file"
# # with np.load("w2v_vectorized_data.npz", allow_pickle=True) as data:
# #     vectorized_data = data['vectorized_data']
# #     mbids = data['mbids']

# kmeans_w2v(vectorized_data, mbids, 32)

# # for k in [16, 32, 64, 128, 256, 512]:
# #     kmeans_w2v(vectorized_data, mbids, k)


# def best_w2v_kmeans():
#     print("finding best w2v kmeans cluster number!")
    
#     top_scores = {}

#     for k in range(10, 800):
#         score = kmeans_w2v(vectorized_data, mbids, k)[1]
#         top_scores[k] = score

#     top_scores = dict(sorted(top_scores.items(), key=lambda item: item[1], reverse=True)[:10])
#     return top_scores

# # run it
# best_w2v_kmeans()

# # 786 w2v clusters gives 0.078949 silhouette score







