from process_text import process_text
from silhouette_best_k import silhouette_best_k
import collections
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

def clusterizer(texts):
    vectorizer = TfidfVectorizer(
        tokenizer=process_text,
        max_df=0.5,
        min_df=1,
        lowercase=True)
    tfidf_model = vectorizer.fit_transform(texts)
    best_k = silhouette_best_k(tfidf_model, len(texts)-2)

    clusterizer = AgglomerativeClustering(n_clusters=best_k, affinity='cosine', linkage='single')
    tfidf_model = vectorizer.fit_transform(texts).todense()
    hac_model = clusterizer.fit_predict(tfidf_model)

    clusters_indexes = collections.defaultdict(list)
    for idx, label in enumerate(hac_model):
        clusters_indexes[label].append(idx)
    clusters_indexes = { cluster_index: txt_indexes for cluster_index, txt_indexes in clusters_indexes.items() }

    print(clusters_indexes)
    return clusters_indexes