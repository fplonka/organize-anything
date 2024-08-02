import sys
import pickle
from html import escape
from openai import OpenAI
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans, BisectingKMeans, OPTICS, Birch
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score 

import os
from typing import List, Dict, Union
from collections import defaultdict
import numpy as np
import umap
import matplotlib.pyplot as plt

import asyncio
import aiohttp

from openai import AsyncOpenAI

from name_clusters import entry_to_string, update_names

reduced_n_components = 30
doing_wikipedia = True

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

import time
from pprint import pprint

def cluster_birch(items: List[str], reduced_embeddings: List[List[float]], 
                      min_cluster_size: int = 5, max_depth: int = 50, current_depth: int = 0) -> Dict:

    if len(items) < min_cluster_size or current_depth >= max_depth:
        return items

    # clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    # clusterer = Birch(n_clusters=None, threshold=0.3)
    clusterer = Birch(n_clusters=100, threshold=0.3)
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    print(cluster_labels)

    clusters = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    for item, embedding, label in zip(items, reduced_embeddings, cluster_labels):
        clusters[label].append(item)
        cluster_embeddings[label].append(embedding)
    
    # return clusters
    
    print("ROOT:")
    pprint(vars(clusterer.root_))
    print("LEAF:")
    pprint(vars(clusterer.dummy_leaf_))
    # time.sleep(30)
    
    cnt = 0
    leaf = clusterer.dummy_leaf_
    while leaf != None:
        leaf = leaf.next_leaf_
        cnt += 1
    
    print("LEAF COUNT:", cnt)
    
    return clusters
        
    print("did clustering.")
    print("have", len(clusters[-1]), "of", len(reduced_embeddings), "as unlabelled..")
    if len(clusters[-1]) == len(reduced_embeddings):
        return items

    if len(clusters[0]) == len(reduced_embeddings):
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

def recursive_cluster(items: List[str], reduced_embeddings: List[List[float]], 
                      min_cluster_size: int = 5, max_depth: int = 50, current_depth: int = 0) -> Dict:
    if len(items) < min_cluster_size or current_depth >= max_depth:
        return items

    # clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    clusterer = OPTICS()
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    print(cluster_labels)

    clusters = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    for item, embedding, label in zip(items, reduced_embeddings, cluster_labels):
        clusters[label].append(item)
        cluster_embeddings[label].append(embedding)
        
    print("did clustering.")
    print("have", len(clusters[-1]), "of", len(reduced_embeddings), "as unlabelled..")
    if len(clusters[-1]) == len(reduced_embeddings):
        return items

    if len(clusters[0]) == len(reduced_embeddings):
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
                              max_k: int = 8, max_depth: int = 10, current_depth: int = 0) -> Dict:
    if len(items) <= 2 or current_depth >= max_depth:
        return items

    best_k = 1
    best_silhouette = -1
    best_dbs = 1000000000000
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
        dbs = davies_bouldin_score(reduced_embeddings, labels) 
        print("indices:", silhouette_avg, dbs)
        if silhouette_avg > best_silhouette:
        # if dbs < best_dbs:
            best_silhouette = silhouette_avg
            best_k = k
            best_dbs = dbs
            best_labels = labels
        prev_silhouttes.append(silhouette_avg)
        if len(prev_silhouttes) >= 4 and prev_silhouttes[-1] < prev_silhouttes[-2] and prev_silhouttes[-2] < prev_silhouttes[-3] and prev_silhouttes[-3] < prev_silhouttes[-4]:
            break

            
    print("best is:", best_silhouette, dbs)
            
    # if (best_silhouette < 0.4 or best_dbs > 1.0) and current_depth != 0:
    #     return items

    if (best_silhouette < 0.2 + current_depth * 0.1) and current_depth != 0:
        return items
    
    # if (best_dbs > 0.9) and current_depth != 0:
    #     return items

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

max_agg_cluster_item_count = 50_000

def agglomerative_cluster(items: List[str], embeddings: List[List[float]]) -> Dict:
    if len(items) > max_agg_cluster_item_count:
        print("have", len(items), "items, doing k=2 k-means")
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        cluster_embeddings = defaultdict(list)
        for item, embedding, label in zip(items, embeddings, labels):
            clusters[label].append(item)
            cluster_embeddings[label].append(embedding)

        result = {}
        for label, cluster_items in clusters.items():
            cluster_name = f"cluster_{label}"
            result[cluster_name] = agglomerative_cluster(cluster_items, cluster_embeddings[label])

        return result

    print("doing aggl clustering on", len(items), "items")
    clusterer = AgglomerativeClustering(distance_threshold=3.0, n_clusters=None)
    clusterer.fit_predict(embeddings)
    
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
                if doing_wikipedia:
                    title, id_ = item
                    html.append(" " * (indent + 2) + f'<li><a href="http://en.wikipedia.org/?curid={escape(id_)}">{escape(title)}</a></li>')
                else:
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
                print("  " * (indent + 1) + str(item))
    # print("  " * indent + "noisy count:", len(clusters.get("unclustered", [])))
    
client = AsyncOpenAI(api_key=os.environ['LS2_OPENAI_KEY'])

from tenacity import retry, stop_after_attempt, wait_random_exponential
@retry(
    stop=stop_after_attempt(500),
    wait=wait_random_exponential(min=2, multiplier=4),
    reraise=True
)
async def get_embeddings_chunk_async(chunk):
    # print("called.. for", chunk[:10])
    print("called..")
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk,
        dimensions=512
    )
    # Sort the entries by their index
    sorted_data = sorted(response.data, key=lambda x: x.index)
    print("got resopnse for chunk!")
    return [entry.embedding for entry in sorted_data]

async def get_embeddings_async(items, chunk_size=2048):
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    tasks = [get_embeddings_chunk_async(chunk) for chunk in chunk_list(items, chunk_size)]
    
    print("have tasks:", tasks)

    chunk_results = await asyncio.gather(*tasks)
    embeddings = [embedding for chunk in chunk_results for embedding in chunk]


    return embeddings

def main(filename: str):
    # TODO: np arrs not lists smh
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
        
        embeddings = asyncio.run(get_embeddings_async(items))

        print("Have:", len(embeddings), "embeddings")

        # Save embeddings to pickle file
        with open(pickle_filename, 'wb') as f:
            pickle.dump((items, embeddings), f)

    print("Embedding count:", len(embeddings))
    
    clusters_pickle_path = "clusters.pickle"
    if os.path.exists(clusters_pickle_path):
        print("LOADED UNNAMED CLUSTERS FROM CACHE")
        with open(clusters_pickle_path, 'rb') as f:
            clusters = pickle.load(f)
    else:
        reduced_embeddings = reduce_dimensions(embeddings)
        print("Reduced embedding shape:", reduced_embeddings.shape)

        # add wikipedia IDS
        if doing_wikipedia:
            with open("data/ids.txt", 'r') as f:
                ids = [line.strip() for line in f]

                # Zip items with their corresponding IDs
                items = list(zip(items, ids))
                print(items[:100])
                print("ZIPPING DONE")

        # clusters = recursive_cluster(items, reduced_embeddings)
        # clusters = cluster_birch(items, reduced_embeddings)
        clusters = agglomerative_cluster(items, reduced_embeddings)
        # clusters = recursive_cluster_k_means(items, reduced_embeddings)
        with open(clusters_pickle_path, 'wb') as f:
            pickle.dump(clusters, f)
    
    print("PROCESSED:")
    update_names(clusters)
    print("RESULT:--------------------------------------------------------------")
    # print_clusters(clusters)

    html = generate_html(clusters)
    with open("output.html", "w") as file: file.write(html)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    main(sys.argv[1])