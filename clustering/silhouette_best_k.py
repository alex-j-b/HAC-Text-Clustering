import gc
import math
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings

def silhouette_calc(n_cluster, tfidf_model):
    '''
    Return the silhouette score for a given number of clusters.
    It's optimised for news articles, you can play with the threshold and metric if the results are not satisfying for your texts' type.
    '''
    print('n_cluster: ', n_cluster)
    km_model = Birch(n_clusters=n_cluster, threshold=0.5, copy=False).fit(tfidf_model)
    label = km_model.labels_
    sil_score = silhouette_score(tfidf_model, label, metric='cosine')
    return sil_score

def silhouette_best_k(tfidf_model, pool, start=2):
    '''
    Find the best k number of clusters based on the highest silhouette score.
    It is optimised to test the least number of k because it can be very long for environments not built for AI.
    Memory is cleared with gc to prevent memory leak.
    '''
    last_digit = str(pool)[len(str(pool))-1]
    temp_scores = {}

    if pool >= 14:
        n_zero_ending = pool if last_digit == '0' else pool - int(last_digit) + 10
        iterator = int(str(n_zero_ending)[:-1])

        check_list = [start+(numb*iterator) if start+(numb*iterator)+iterator < start+pool else start+pool for numb in range(0,10)]

        middle = math.ceil(len(check_list)/2)
        middle_sil = silhouette_calc(check_list[middle], tfidf_model)
        temp_scores[check_list[middle]] = middle_sil
        before_loop = reversed(check_list[:middle])
        after_loop = check_list[middle+1:]

        warnings.simplefilter("error", ConvergenceWarning)
        for idx, n_cluster in enumerate(before_loop):
            try:
                sil_score = silhouette_calc(n_cluster, tfidf_model)
                temp_scores[n_cluster] = sil_score
                if any(value > sil_score for key, value in temp_scores.items()):
                    break
            except ConvergenceWarning:
                warnings.simplefilter("ignore", ConvergenceWarning)
                sil_score = silhouette_calc(n_cluster, tfidf_model)
                temp_scores[n_cluster] = sil_score
                check_list = check_list[:idx+1]
                break
            gc.collect()
            del gc.garbage[:]

        warnings.simplefilter("error", ConvergenceWarning)
        for idx, n_cluster in enumerate(after_loop):
            try:
                sil_score = silhouette_calc(n_cluster, tfidf_model)
                temp_scores[n_cluster] = sil_score
                if any(value > sil_score for key, value in temp_scores.items()):
                    break
            except ConvergenceWarning:
                warnings.simplefilter("ignore", ConvergenceWarning)
                sil_score = silhouette_calc(n_cluster, tfidf_model)
                temp_scores[n_cluster] = sil_score
                check_list = check_list[:idx+1]
                break
            gc.collect()
            del gc.garbage[:]

        temp_scores = dict(sorted(temp_scores.items()))
        all_scores = [value for key, value in temp_scores.items()]
        all_k = [key for key, value in temp_scores.items()]

        best_score = max(all_scores)
        best_k_idx = all_scores.index(best_score)
        best_k = all_k[best_k_idx]

        next_start = best_k if best_k_idx == 0 else all_k[best_k_idx-1] + 1
        max_k = best_k if best_k_idx == len(all_k)-1 else all_k[best_k_idx+1] - 1
        next_pool = max_k - next_start

        gc.collect()
        del gc.garbage[:]

        return silhouette_best_k(tfidf_model, next_pool, next_start)

    else:
        middle = math.ceil(len(range(start,start+pool+1))/2)
        middle_sil = silhouette_calc(range(start,start+pool+1)[middle], tfidf_model)
        temp_scores[range(start,start+pool+1)[middle]] = middle_sil
        before_loop = reversed(range(start,start+pool+1)[:middle])
        after_loop = range(start,start+pool+1)[middle+1:]

        for n_cluster in before_loop:
            sil_score = silhouette_calc(n_cluster, tfidf_model)
            if any(value > sil_score for key, value in temp_scores.items()):
                break
            else:
                temp_scores[n_cluster] = sil_score
            gc.collect()
            del gc.garbage[:]

        for n_cluster in after_loop:
            sil_score = silhouette_calc(n_cluster, tfidf_model)
            if any(value > sil_score for key, value in temp_scores.items()):
                break
            else:
                temp_scores[n_cluster] = sil_score
            gc.collect()
            del gc.garbage[:]

        all_scores = [value for key, value in temp_scores.items()]
        all_k = [key for key, value in temp_scores.items()]

        best_score = max(all_scores)
        best_k_idx = all_scores.index(best_score)
        best_k = all_k[best_k_idx]

        print("BEST_K: ", best_k, " WITH SCORE: ", best_score)
        return best_k