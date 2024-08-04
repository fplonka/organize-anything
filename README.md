## Description

organize-anything is a Python script which uses machine learning to organize a list of text items into named categories and subcategories. You can pretty-print the output, generate HTML from it, or just work with the tree structure in code.

See the [demo page](https://fplonka.dev/organize-anything) showing a million Wikipedia articles categorized with organize-anything for an example. 

## How it works

Each text item is turned into a vector embedding (using OpenAI's embedding API). [UMAP](https://umap-learn.readthedocs.io/en/latest/) is ran on these vectors to reduce them to 30 dimensions, and these reduced embeddings are hierarchically organized using [agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html). The subtrees created this way are then fed bottom-up to an LLM (gpt-4o-mini) to give them informative names.

Since agglomerative clustering is slow, organize-anything cheats slightly and recursively runs [k-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) with k=2 to break up the data first. This only matters for quite large datasets (>50,000 items). 

I experimented with some other methods (HDBSCAN, OPTICS, recursively running k-means) but I found the results inferior.

## Usage

Copy the script and install the dependencies in requirements.txt. The input format is a file with one text item per line. 

Make sure you either provide an OpenAI API key with `--api-key API_KEY` or set the `OPENAI_API_KEY` env var. As a reference, running organize-anything for 1 million wikipedia article titles cost ~$2 for the gpt4o-mini calls and ~$0.10 for the vector embedding calls. 

```
usage: script.py [-h] [--api-key API_KEY] [--generate-html] [--dont-name] [--verbose]
                 [--embedding-model EMBEDDING_MODEL] [--naming-model NAMING_MODEL]
                 [--max-cluster-size MAX_CLUSTER_SIZE] [--rate-limit RATE_LIMIT]
                 input_file

Organize a list of items into categories and subcategories.

positional arguments:
  input_file            Input file containing items to cluster

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     OpenAI API key; required if OPENAI_API_KEY is not set (default: None)
  --generate-html       Generate HTML output instead of pretty-printed text (default: False)
  --dont-name           Skip the LLM category and subcategory naming step (default: False)
  --verbose             Print verbose output (default: False)
  --embedding-model EMBEDDING_MODEL
                        Model to use for embeddings (default: text-embedding-3-small)
  --naming-model NAMING_MODEL
                        Model to use for naming clusters (default: gpt-4o-mini)
  --max-cluster-size MAX_CLUSTER_SIZE
                        Maximum size of a data subset on which we run agglomerative clustering
                        (larger chunks are first split up with k-means) (default: 50000)
  --rate-limit RATE_LIMIT
                        Rate limit for API calls (per minute) (default: 250)
```
