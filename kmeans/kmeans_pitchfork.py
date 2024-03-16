"""
kmeans_pitchfork.py
Building semantic clusters with k-means algorithm on pitchfork review data.
8 March 2024
"""

### (0) libraries to import ###
import json
from tqdm import tqdm
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np 
from transformers import GPT2Tokenizer, GPT2Model
import pickle
import joblib
import torch


### (1) data preprocessing and loading ###
"""
We have data in /pitchfork_midis with md5.mid and md5.txt.
Need to convert each md5.txt into a dictionary {md5 : "whole doc string"}
"""

def load_data(data_path):
    """
    load and preprocess the data.

    params: 
    data_path: folder /pitchfork_midis
    
    return:
    md5_to_text (dict): {md5 : "whole doc string"}
    """

    # initialize return dictionary
    md5_to_text = {}

    # loop over every file in data directory
    for filename in os.listdir(data_path):
       
        # ensure that file is a txt file and not a mid
        if filename.endswith(".txt"):

            # md5 is filename minus '.txt'
            md5 = filename[:-4]

            # whole doc string is the actual contents of the file
            with open(f"pitchfork_midis/{filename}", 'rb') as file:

                # turn byte stream into string and drop the first (b') and last (')
                text = str(file.read())[2:-1]

            # place text in dictionary as value to key md5
            md5_to_text[md5] = text

    # return the dict
    return md5_to_text



### (2) GPT-2 NN approach to feed into kmeans ###
#  -  empirically this seems to be the best approach
#  -  also the most computationally expensive


# (a) load data from saved when possible 
def load_GPT_vectorized_data(file_path):
    """
    Load the saved GPT2 vectorized data.

    Params:
    file_path (str): Path to the saved npz file containing vectorized data.

    Returns:
    embeddings: numpy array of GPT2 embeddings.
    md5s: numpy array of md5s.
    """

    print(f"loading GPT vectorized data from {file_path}")
    # Load the npz file
    loaded_data = np.load(file_path)

    # Extract embeddings and md5
    embeddings = loaded_data['vectorized_data']
    md5s = loaded_data['md5s']

    return embeddings, md5s


# (b) need a function to tokenize and vectorize the data
def GPT_vectorize(data_preproc, save_file=None):
    """
    clean, lowercase, tokenize etc. the initially preprocessed data

    params:
    data_preproc (dict): {md5: "whole-wikidoc string"}

    returns:
    embeddings to use in kmeans
    """

    # if we have already run it, just open the file
    if save_file:
        embeddings, md5s = load_GPT_vectorized_data(save_file)
        print("loaded GPT vectorized data!")

    # otherwise, run the GPT approach
    else:

        # load pretrained GPT tokenizer and model
        print("loading GPT tokenizer and model")
        GPT_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding=True)
        GPT_tokenizer.pad_token = '[PAD]'
        GPT_model = GPT2Model.from_pretrained('gpt2')
        print("loaded!")

        # initialize lists to store embeddings, md5
        embeddings = []
        md5s = []

        # iterate over preprocessed data and get embeddings
        print("iterating...")
        iter_count = 0
        for md5, doc in tqdm(data_preproc.items()):

            # just to keep track of where we are
            iter_count += 1
            if iter_count % 1000 == 0:
                print(f"iteration number {iter_count}")
                print(f"    currently working on {md5} : {doc[:20]}...")

            # tokenize the doc text, get pytorch tensors
            encoded_input = GPT_tokenizer(doc, return_tensors="pt", padding=True, truncation=True)

            # generate the embeddings
            with torch.no_grad():

                # get outputs from GPT model
                outputs = GPT_model(**encoded_input)

                # get the [CLS] (classification) token for sequence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            # append the embedding and corresponding md5 to the lists
            embeddings.append(cls_embedding)
            md5s.append(md5)

        # convert the lists of embeddings and md5s to numpy arrays
        embeddings = np.array(embeddings)
        md5s = np.array(md5s)

        # save the vectorized data and corresponding md5s to a file
        np.savez('GPT_vectorized_data.npz', vectorized_data=embeddings, md5s=md5s)

    # compress the embeddings so that kmeans will work
    embeddings = np.concatenate(embeddings, axis=0)

    print("data GPT embedded!")
    return embeddings, md5s


# (c) run kmeans with GPT
def kmeans_GPT(embeddings, md5s, K, save=False):

    # set and run kmeans model
    kmeans = KMeans(n_clusters=K)
    print(f"running kmeans on {K} clusters...")
    kmeans.fit(embeddings)

    # get cluster assignments for md5s 
    cluster_assignments = kmeans.labels_
    md5_clusters = {md5: cluster for md5, cluster in zip(md5s, cluster_assignments)}

    # get silhouette scores
    silhouette = silhouette_score(embeddings, cluster_assignments)
    print(f"silhouette score for {K} clusters: {silhouette}")

    # save model and clusters
    if save:
        joblib.dump(kmeans, f'GPT_{K}k_means_model.joblib')
        save_clusters(md5_clusters, f"md5_GPT_{K}_clusters.pickle")

    return md5_clusters, silhouette


# (d) save the clusters for later
def save_clusters(clusters, filename):
    """
    Save clusters and corresponding documents to a pickle file.

    Parameters:
    clusters (dict): Dictionary containing clusters {cluster : [(md5, "whole-wikidoc string")]}.
    filename (str): Name of the pickle file to save.
    """
    with open(filename, 'wb') as f:
        pickle.dump(clusters, f)



### (3) stack of function calls ###

# (a) set data path
data_path = "pitchfork_midis"

# (b) load and slightly preprocess the data
data = load_data(data_path)
print(len(data.keys()))  # sanity check

# (c) feed loaded data into vectorizer
embeddings, md5s = GPT_vectorize(data_preproc=data, save_file="GPT_vectorized_data.npz")

# (d) try different k values
for k_val in [8, 16, 32, 64, 128, 256, 512]:
    clusters, silhouette = kmeans_GPT(embeddings=embeddings, md5s=md5s, K=k_val, save=False)

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
    print(f"Mean number of md5s per cluster: {mean_assigned}")
    print(f"Standard deviation of number of md5s per cluster: {std_assigned}")

# try with 256, which is empirically the best
kmeans_GPT(embeddings=embeddings, md5s=md5s, K=256, save=True)
