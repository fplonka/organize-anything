import os
import sys
import json
import asyncio
import random
from typing import List, Dict, Union
from html import escape

import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
import umap
from tenacity import retry, stop_after_attempt, wait_random_exponential
from aiolimiter import AsyncLimiter

class HierarchicalClusterer:
    def __init__(self, api_key: str, verbose: bool = False, max_cluster_size: int = 50000,
                 rate_limit: int = 2000, embedding_model: str = "text-embedding-3-small",
                 naming_model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.verbose = verbose
        self.max_cluster_size = max_cluster_size
        self.limiter = AsyncLimiter(rate_limit, 60)  # rate_limit requests per minute
        self.embedding_model = embedding_model
        self.naming_model = naming_model

    async def get_embeddings(self, items: List[str], chunk_size: int = 2048) -> List[List[float]]:
        """
        Asynchronously get embeddings for a list of items using the OpenAI API.
        
        This method chunks the input to avoid hitting API limits and uses retries
        to handle potential API errors.
        """
        @retry(stop=stop_after_attempt(500), wait=wait_random_exponential(min=2, multiplier=4), reraise=True)
        async def get_embeddings_chunk(chunk):
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=chunk,
                dimensions=512
            )
            return [entry.embedding for entry in sorted(response.data, key=lambda x: x.index)]

        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        tasks = [get_embeddings_chunk(chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        return [embedding for chunk in chunk_results for embedding in chunk]

    def reduce_dimensions(self, embeddings: List[List[float]], n_components: int = 30) -> np.ndarray:
        """
        Reduce the dimensionality of the embeddings using UMAP.
        """
        reducer = umap.UMAP(n_components=n_components, metric='cosine')
        return reducer.fit_transform(embeddings)

    def cluster(self, items: List[str], embeddings: List[List[float]]) -> Dict:
        """
        Perform hierarchical clustering on the items based on their embeddings.
        
        For large datasets, this method first uses k-means to split the data
        into smaller clusters, then applies agglomerative clustering to each subset.
        
        The resulting cluster structure is a nested dictionary where:
        - Keys are cluster names (initially generic names like "cluster_0")
        - Values are either:
          - A list of items (for leaf nodes)
          - Another dictionary representing a sub-cluster (for internal nodes)
        """
        if len(items) > self.max_cluster_size:
            if self.verbose:
                print(f"Performing k-means clustering with k=2 on {len(items)} items")
            kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            clusters = {}
            for label in range(2):
                cluster_items = [item for item, l in zip(items, labels) if l == label]
                cluster_embeddings = [emb for emb, l in zip(embeddings, labels) if l == label]
                clusters[f"cluster_{label}"] = self.cluster(cluster_items, cluster_embeddings)
            return clusters

        if self.verbose:
            print(f"Performing agglomerative clustering on {len(items)} items")
        clusterer = AgglomerativeClustering(distance_threshold=3.0, n_clusters=None)
        clusterer.fit(embeddings)
        return self.build_hierarchy(items, clusterer.children_, clusterer.distances_)

    def build_hierarchy(self, items: List[str], children: List[List[int]], distances: List[float], merge_threshold: float = 3.0) -> Dict:
        """
        Build a hierarchical structure from the results of agglomerative clustering.
        
        This method creates a tree-like structure where each node is either a leaf
        (containing a list of items) or an internal node (containing sub-clusters).
        See the sklearn docs for details about what children and distances mean. A 
        lower merge_threshold means different clusters are merged more eagerly, so
        the result will have fewer, larger clusters.
        """
        n_samples = len(items)
        n_nodes = n_samples + len(children)
        nodes = [{"name": f"cluster_{i}", "content": None} for i in range(n_nodes)]
        
        for i in range(n_samples):
            nodes[i]["content"] = [items[i]]
        
        for i, ((left, right), distance) in enumerate(zip(children, distances)):
            parent = n_samples + i
            left_content, right_content = nodes[left]["content"], nodes[right]["content"]
            
            if distance < merge_threshold:
                nodes[parent]["content"] = left_content + right_content
            else:
                nodes[parent]["content"] = {
                    nodes[left]["name"]: left_content,
                    nodes[right]["name"]: right_content
                }
        
        return nodes[-1]["content"]

    async def update_names(self, entry: Union[Dict, list]):
        """
        Recursively update the names of clusters using the LLM.
        
        This method traverses the cluster hierarchy, replacing generic cluster
        names with more descriptive names generated by the language model.
        """
        if isinstance(entry, list):
            return

        if isinstance(entry, dict):
            keys = list(entry.keys())
            await asyncio.gather(*[self.update_names(entry[k]) for k in keys])
            
            async def process_key(k):
                return await self.get_name_with_llm(entry[k])
            
            new_names = await asyncio.gather(*[process_key(k) for k in keys])
            for old_key, new_key in zip(keys, new_names):
                if old_key in entry:
                    entry[new_key] = entry.pop(old_key)

    @retry(stop=stop_after_attempt(500), wait=wait_random_exponential(min=2, multiplier=4), reraise=True)
    async def get_name_with_llm(self, entry: Union[Dict, list]) -> str:
        """
        Generate a descriptive name for a cluster using the language model.
        
        This method constructs a prompt based on the cluster contents and uses
        the OpenAI API to generate a suitable name.
        """
        entry_string = self.entry_to_string(entry)
        prompt = (
            "You are an expert at categorization and naming. Given the following data, "
            "provide a simple and descriptive name for the category that groups all these items together. "
            "The name should be concise yet informative, capturing the essence of the category. "
            f"Here's the data:\n\n{entry_string}\n\n"
            "Respond with a JSON object in the format: { \"name\": \"Your Category Name\" }"
        )

        async with self.limiter:
            completion = await self.client.chat.completions.create(    
                model=self.naming_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in categorization and naming."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                max_tokens=30
            )

        try:
            response_json = json.loads(completion.choices[0].message.content)
            return response_json["name"]
        except (json.JSONDecodeError, KeyError):
            return "Unnamed Category"

    def entry_to_string(self, entry: Union[Dict, list], indent: int = 0, max_items: int = 100) -> str:
        """
        Convert a cluster entry (either a dict or a list) to a string representation.
        
        This is used to create a human-readable format for the LLM prompt.
        """
        result = []
        if isinstance(entry, dict):
            for key, value in entry.items():
                result.append("  " * indent + str(key))
                result.append(self.entry_to_string(value, indent + 1, max_items))
        else:
            sample_size = min(max_items, len(entry))
            for item in random.sample(entry, sample_size):
                result.append("  " * indent + str(item))
        return "\n".join(result)

    def generate_html(self, clusters: Dict) -> str:
        """
        Generate an HTML representation of the cluster hierarchy.
        
        This method creates an expandable tree view of the clusters using HTML
        details and summary tags.
        """
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
        
        content = create_list(clusters)
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

    def print_clusters(self, clusters: Dict, indent: int = 0):
        """
        Print a human-readable representation of the cluster hierarchy.
        """
        lines = []
        for key, value in clusters.items():
            print("  " * indent + str(key))
            if isinstance(value, dict):
                lines.append(self.print_clusters(value, indent + 1))
                # self.print_clusters(value, indent + 1)
            else:
                for item in value:
                    lines.apend("  " * (indent + 1) + str(item))
                    # print("  " * (indent + 1) + str(item))
        return "\n".join(lines)

    async def cluster_and_name_async(self, items: List[str], dont_name: bool = False) -> Dict:
        """
        Perform the entire clustering and naming process.
        
        """
        print(f"Getting embeddings for {len(items)} items...")
        embeddings = await self.get_embeddings(items)
        
        print("Reducing dimensions...")
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        print("Clustering...")
        clusters = self.cluster(items, reduced_embeddings)
        
        if not dont_name:
            print("Naming clusters...")
            await self.update_names(clusters)
        
        return clusters

    def cluster_and_name(self, items: List[str], dont_name: bool = False) -> Dict:
        """
        Synchronous wrapper for the cluster_and_name_async method.
        """
        return asyncio.run(self.cluster_and_name_async(items, dont_name))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Organize a list of items into categories and subcategories.", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help="Input file containing items to cluster")
    parser.add_argument("--api-key", help="OpenAI API key; required if OPENAI_API_KEY is not set")
    parser.add_argument("--generate-html", action="store_true", help="Generate HTML output instead of pretty-printed text")
    parser.add_argument("--dont-name", action="store_true", help="Skip the LLM category and subcategory naming step")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Model to use for embeddings")
    parser.add_argument("--naming-model", default="gpt-4o-mini", help="Model to use for naming clusters")
    parser.add_argument("--max-cluster-size", type=int, default=50000, help="Maximum size of a data subset on which we run agglomerative clustering (larger chunks are first split up with k-means)")
    parser.add_argument("--rate-limit", type=int, default=250, help="Rate limit for API calls (per minute)")
    args = parser.parse_args()

    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key not provided. Run with --api-key KEY or set the OPENAI_API_KEY env variable.")
            sys.exit(1)

    clusterer = HierarchicalClusterer(
        api_key, 
        verbose=args.verbose, 
        max_cluster_size=args.max_cluster_size,
        rate_limit=args.rate_limit,
        embedding_model=args.embedding_model,
        naming_model=args.naming_model
    )

    with open(args.input_file, 'r') as f:
        items = [line.strip() for line in f]

    print(f"Clustering {len(items)} items...")
    clusters = clusterer.cluster_and_name(items, args.dont_name)

    if args.generate_html:
        output = clusterer.generate_html(clusters)
    else:
        output = clusterer.print_clusters(clusters)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()