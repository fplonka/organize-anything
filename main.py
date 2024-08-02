import sys
import pickle
from html import escape
from openai import OpenAI
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.metrics import silhouette_score
import os
from typing import List, Dict, Union
from collections import defaultdict
import numpy as np
import umap
import matplotlib.pyplot as plt

from name_clusters import entry_to_string, update_names

reduced_n_components = 30

def reduce_dimensions(embeddings, n_components=reduced_n_components):
    reducer = umap.UMAP(n_components=n_components, metric='cosine')
    return reducer.fit_transform(embeddings)

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

def reduce_dimensions_mds(embeddings, n_components=reduced_n_components):
    # Compute cosine distances
    distances = cosine_distances(embeddings)
    
    # Create MDS object
    mds = MDS(n_components=n_components, 
              dissimilarity='precomputed', 
              random_state=42, 
              n_jobs=-1)  # Use all available cores
    
    # Fit and transform the data
    reduced_embeddings = mds.fit_transform(distances)
    
    return reduced_embeddings

def visualize_embeddings(reduced_embeddings, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def recursive_cluster(items: List[str], reduced_embeddings: List[List[float]], 
                      min_cluster_size: int = 5, max_depth: int = 10, current_depth: int = 0) -> Dict:
    if len(items) < min_cluster_size or current_depth >= max_depth:
        return items

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    clusters = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    for item, embedding, label in zip(items, reduced_embeddings, cluster_labels):
        clusters[label].append(item)
        cluster_embeddings[label].append(embedding)
        
    print("did clustering.")
    print("have", len(clusters[-1]), "of", len(reduced_embeddings))
    if len(clusters[-1]) == len(reduced_embeddings):
        return items

    result = {}
    for label, cluster_items in clusters.items():
        cluster_name = f"cluster_{label}"
        if label == -1:
            cluster_name = "unclustered"

        result[cluster_name] = recursive_cluster(
            cluster_items, 
            cluster_embeddings[label], 
            min_cluster_size, 
            max_depth, 
            current_depth + 1
        )

    return result

def recursive_cluster_k_means(items: List[str], reduced_embeddings: List[List[float]], 
                              max_k: int = 20, max_depth: int = 10, current_depth: int = 0) -> Dict:
    if len(items) <= 2 or current_depth >= max_depth:
        return items

    best_k = 1
    best_silhouette = -1
    best_labels = None
    
    if max_k >= len(items):
        max_k = len(items) - 1
        
    prev_silhouttes = []

    for k in range(2, max_k + 1):
        print("trying k=", k)
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        # kmeans = BisectingKMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(reduced_embeddings)
        silhouette_avg = silhouette_score(reduced_embeddings, labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_k = k
            best_labels = labels
        prev_silhouttes.append(silhouette_avg)
        if len(prev_silhouttes) >= 4 and prev_silhouttes[-1] < prev_silhouttes[-2] and prev_silhouttes[-2] < prev_silhouttes[-3] and prev_silhouttes[-3] < prev_silhouttes[-4]:
            break

            
    # print("best is:", best_silhouette)
            
    if best_silhouette < 0.6 and current_depth != 0:
        return items

    if best_labels is None:
        return items
    
    # if best_k == 1:
        # return items

    clusters = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    for item, embedding, label in zip(items, reduced_embeddings, best_labels):
        clusters[label].append(item)
        cluster_embeddings[label].append(embedding)
    
    print(f"K-Means clustering at depth {current_depth}, best_k: {best_k}, silhouette: {best_silhouette}")

    result = {}
    for label, cluster_items in clusters.items():
        cluster_name = f"cluster_{label}"
        result[cluster_name] = recursive_cluster_k_means(
            cluster_items, 
            cluster_embeddings[label], 
            max_k, 
            max_depth, 
            current_depth + 1
        )

    return result

def agglomerative_cluster(items: List[str], embeddings: List[List[float]]) -> Dict:
    # clusterer = AgglomerativeClustering(n_clusters=100)
    clusterer = AgglomerativeClustering(distance_threshold=3.0, n_clusters=None)
    # clusterer = AgglomerativeClustering(distance_threshold=1.5, n_clusters=None, metric='cosine', linkage='single')
    # clusterer = AgglomerativeClustering(n_clusters=50, metric='cosine', linkage='average')
    cluster_labels = clusterer.fit_predict(embeddings)
    
    print(clusterer.children_)
    print("DISTANCES:", clusterer.distances_)
    
    return build_hierarchy(items, clusterer.children_, clusterer.distances_)

def build_hierarchy(items: List[str], children: List[List[int]], distances: List[float], merge_threshold: float = 3.0) -> Dict:
    n_samples = len(items)
    n_nodes = n_samples + len(children)
    
    # Initialize all nodes
    nodes = [{"name": f"cluster_{i}", "content": None} for i in range(n_nodes)]
    
    # Set leaf nodes
    for i in range(n_samples):
        nodes[i]["content"] = [items[i]]
    
    # Build the tree bottom-up
    for i, ((left, right), distance) in enumerate(zip(children, distances)):
        parent = n_samples + i
        
        left_content = nodes[left]["content"]
        right_content = nodes[right]["content"]
        
        if distance < merge_threshold:
            # Merge the contents if the distance is below the threshold
            if isinstance(left_content, list) and isinstance(right_content, list):
                nodes[parent]["content"] = left_content + right_content
            else:
                raise Exception("WEIRD??")
        else:
            # Keep the structure if the distance is above the threshold
            nodes[parent]["content"] = {
                nodes[left]["name"]: left_content,
                nodes[right]["name"]: right_content
            }
    
    # The last node is the root
    return nodes[-1]["content"]

def generate_html(clusters: Dict, indent: int = 0) -> str:
    def create_list(items: Union[Dict, list], indent: int = 0) -> str:
        html = []
        if isinstance(items, dict):
            for key, value in items.items():
                html.append(" " * indent + f"<details>")
                html.append(" " * (indent + 2) + f"<summary>{escape(str(key))}</summary>")
                html.append(create_list(value, indent + 4))
                html.append(" " * indent + f"</details>")
        else:
            html.append(" " * indent + "<ul>")
            for item in items:
                html.append(" " * (indent + 2) + f"<li>{escape(str(item))}</li>")
            html.append(" " * indent + "</ul>")
        return "\n".join(html)
    
    content = create_list(clusters, indent)
    
    return f"""
    <html>
    <head>
        <style>
            details>*:not(summary):not(ul) {{
                margin-left: 2em;
            }}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """

def print_clusters(clusters: Dict, indent: int = 0):
    for key, value in clusters.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):
            print_clusters(value, indent + 1)
        else:
            for item in value:
                print("  " * (indent + 1) + item)
    # print("  " * indent + "noisy count:", len(clusters.get("unclustered", [])))

def main(filename: str):
    base_filename = os.path.splitext(filename)[0]
    pickle_filename = base_filename + '.pickle'
    
    if os.path.exists(pickle_filename):
        print(f"Loading items and embeddings from {pickle_filename}")

        with open(pickle_filename, 'rb') as f:
            items, embeddings = pickle.load(f)

    else:
        with open(filename, 'r') as f:
            items = [line.strip() for line in f]
        
        print("Have", len(items), "items")
        
        def chunk_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
    
        client = OpenAI(api_key=os.environ['LS2_OPENAI_KEY'])
        embeddings = []

        for chunk in chunk_list(items, 2048):
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.extend([entry.embedding for entry in response.data])

            print("Have:", len(embeddings), "embeddings")
        
        # TODO: responses could be out of order..?

        # Save embeddings to pickle file
        with open(pickle_filename, 'wb') as f:
            pickle.dump((items, embeddings), f)


    print("Embedding count:", len(embeddings))
    
    clusters_pickle_path = "clusters.pickle"
    if os.path.exists(clusters_pickle_path) and False:
        print("LOADED UNNAMED CLUSTERS FROM CACHE")
        with open(clusters_pickle_path, 'rb') as f:
            clusters = pickle.load(f)
    else:
        reduced_embeddings = reduce_dimensions(embeddings)
        print("Reduced embedding shape:", reduced_embeddings.shape)

        # clusters = recursive_cluster(items, reduced_embeddings)
        clusters = agglomerative_cluster(items, reduced_embeddings)
        # clusters = recursive_cluster_k_means(items, reduced_embeddings)
        with open(clusters_pickle_path, 'wb') as f:
            pickle.dump(clusters, f)
    
    print("PROCESSED:")
    # update_names(clusters)
    print("RESULT:--------------------------------------------------------------")
    print_clusters(clusters)

    html = generate_html(clusters)
    with open("output.html", "w") as file: file.write(html)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    main(sys.argv[1])