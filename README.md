## Description

organize-anything is a Python script which uses machine learning to organize a list of text items into named categories and subcategories. You can pretty-print the output, generate HTML from it, or just work with the tree structure in code.

See the [demo page](https://fplonka.dev/organize-anything) showing a million Wikipedia articles categorized with organize-anything for an example. 

## How it works

Each text item is turned into a vector embedding (using OpenAI's embedding API). [UMAP](https://umap-learn.readthedocs.io/en/latest/) is ran on these vectors to reduce them to 30 dimensions, and these reduced embeddings are hierarchically organized using [agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html). The subtrees created this way are then fed bottom-up to an LLM (gpt-4o-mini) to give them informative names.

Since agglomerative clustering is slow, organize-anything cheats slightly and recursively runs [k-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) with k=2 to break up the data first. This only matters for quite large datasets (>50,000 items).  

## Usage

TODO: clean up code (separate out wikipedia specific bs) and expose some kind of sane api, script flags/params, reqs.txt, download/running example

imagining something like `python main.py file.txt --generate-html -o out.html`
