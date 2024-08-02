# code for naming the clusters

import sys
import pickle
from html import escape
from openai import OpenAI
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import os
from typing import List, Dict, Union
from collections import defaultdict
import numpy as np
import umap
import matplotlib.pyplot as plt
from random import sample
import random
import json

import asyncio
from openai import AsyncOpenAI
import openai


max_items_to_show = 100
def entry_to_string(entry: Union[Dict, list], indent: int = 0) -> str:
    result = ""
    
    if isinstance(entry, dict):
        for key, value in entry.items():
            result += "  " * indent + str(key) + "\n"
            flattened_items = flatten_items(value)
            sample_size = min(max_items_to_show, len(flattened_items))
            sampled_items = random.sample(flattened_items, sample_size)
            for item in sampled_items:
                result += "  " * (indent + 1) + item + "\n"
    else:
        if len(entry) > max_items_to_show:
            entry = random.sample(entry, max_items_to_show)
        for item in entry:
            result += "  " * indent + item + "\n"
    
    return result

def flatten_items(entry: Union[Dict, list]) -> list:
    if isinstance(entry, dict):
        items = []
        for value in entry.values():
            items.extend(flatten_items(value))
        return items
    else:
        return entry

async def update_names_async(entry: Union[Dict, list]):
    if isinstance(entry, list):
        return  # Do nothing for lists
    
    if isinstance(entry, dict):
        # First, recursively update names for all nested entries
        keys = list(entry.keys())
        
        async def update_nested(k):
            await update_names_async(entry[k])

        nested_tasks = [update_nested(k) for k in keys]
        await asyncio.gather(*nested_tasks)

        async def process_key(k):
            print("UNDER ME:")
            print(entry_to_string(entry[k]))
            return await get_name_with_llm_async(entry[k])
        
        tasks = [process_key(k) for k in keys]
        results = await asyncio.gather(*tasks)
        m = dict(zip(keys, results))
        
        # Update the keys based on the new names
        for old_key, new_key in m.items():
            if old_key in entry:
                entry[new_key] = entry.pop(old_key)

# This is a synchronous wrapper for the asynchronous function
def update_names(entry: Union[Dict, list]):
    asyncio.run(update_names_async(entry))

client = AsyncOpenAI(api_key=os.environ['LS2_OPENAI_KEY'])

from aiolimiter import AsyncLimiter
rate_limit_per_minute = 2000
limiter = AsyncLimiter(rate_limit_per_minute, 60)
# limiter = AsyncLimiter(1, 1)

from tenacity import retry, stop_after_attempt, wait_random_exponential
@retry(
    stop=stop_after_attempt(500),
    wait=wait_random_exponential(min=2, multiplier=4),
    reraise=True
)
async def get_name_with_llm_async(entry: Union[Dict, list]) -> str:
    entry_string = entry_to_string(entry)
    
    if isinstance(entry, dict):
        prompt = (
            "You are an expert at categorization and naming. Given the following hierarchical data, "
            "provide a simple and descriptive name for the category that groups all these subcategories together. "
            "The name should be concise yet informative, capturing the essence of the category. "
            "Here's the data:\n\n"
            f"{entry_string}\n\n"
            "Respond with a JSON object in the format: { \"name\": \"Your Category Name\" }"
        )
    else:  # list
        prompt = (
            "You are an expert at categorization and naming. Given the following list of items, "
            "provide a simple and descriptive name for the category that groups all these items together. "
            "The name should be concise yet informative, capturing the essence of the category. "
            "Here's the list:\n\n"
            f"{entry_string}\n\n"
            "Respond with a JSON object in the format: { \"name\": \"Your Category Name\" }"
        )

    
    async with limiter:
        try:
            completion = await client.chat.completions.create(    
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in categorization and naming."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                max_tokens=30
            )
        except Exception as e:
            print(get_name_with_llm_async.retry.statistics)
            print("error:", e)
            raise(e)

        try:
            response_content = completion.choices[0].message.content
            response_json = json.loads(response_content)
            print("GOT:", response_json["name"])
            return response_json["name"]
        except (json.JSONDecodeError, KeyError):
            return "Unnamed Category"