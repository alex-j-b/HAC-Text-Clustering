# HAC-Text-Clustering
It is the HAC algorithm I use to sort newspaper articles by news. You can adapt it to pretty much any type of text.
HAC means "Hierarchical Agglomerative Clustering", it worked out better for me than KMeans.
It use the silhouette score to find the best k.


Table of Contents 
---------------------------
process_text.py

	Tokenize and Stem data to get meaningful list of words in regard to the clusterizer
	
silhouette_best_k.py

	Find the best k number of clusters based on the highest silhouette score.

main.py

    Find the best k with silhouette score
	Apply Hierarchical clustering from the best k
	Return clusters


You might need to adapt few parameters to your type of dataset. Here are some changes that you can try :
<br/>
process_text.py  
Add more stopwords.  
Replace or remove the stemmer.  
<br/>
silhouette_best_k.py  
Change the Birch threshold  
Change the silhouette_score metric  
<br/>
main.py  
Change TfidfVectorizer max_df and/or min_df.  
Try others AgglomerativeClustering affinity and linkage options.  
Replace AgglomerativeClustering by KMeans.  

Project Requirements
----------------------------

python 3
pip install requirements.txt
(nltk, scikit_learn)