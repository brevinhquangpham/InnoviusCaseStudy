from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer

from text_preprocessing_helpers import text_preprocessing_pipeline

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tqdm.pandas()


def ingest_data(file_path):
    df = pd.read_excel(file_path)
    return df


def split_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]


def process_chunk(chunk_data):
    chunk, text_columns, batch_size = chunk_data
    # Initialize the model and tokenizer inside the worker process
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Concatenate text columns
    texts = chunk[text_columns].fillna("").agg(" ".join, axis=1).tolist()

    # Preprocess all texts
    processed_texts = [text_preprocessing_pipeline(text) for text in texts]

    # Process in batches
    embeddings = []
    for i in tqdm(
        range(0, len(processed_texts), batch_size),
        desc="Processing batches",
        position=0,
        leave=False,
    ):
        batch_texts = processed_texts[i : i + batch_size]
        # Encode batch of texts at once
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings.extend(batch_embeddings)

    chunk["embeddings"] = embeddings
    return chunk


def calculate_embedding(text, tokenizer, model):
    tokens = tokenizer.tokenize(text)
    max_length = 512
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]

    chunk_embeddings = [
        model.encode(tokenizer.convert_tokens_to_string(chunk)) for chunk in chunks
    ]

    return np.mean(chunk_embeddings, axis=0)


# Main function to parallelize embedding calculation
def get_embeddings(df, text_columns, n_workers=4, batch_size=32):
    # Split the DataFrame into chunks
    chunks = np.array_split(df, n_workers)
    # Create tuple of (chunk, text_columns, batch_size) for each worker
    chunk_data = [(chunk, text_columns, batch_size) for chunk in chunks]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_chunk, chunk_data),
                total=n_workers,
                desc="Processing chunks in parallel",
            )
        )

    return pd.concat(results, ignore_index=True)


def preprocessing(df):
    df = df.drop_duplicates()
    columns_to_remove_non_strings = [
        "Description",
        "Description.1",
        "Sourcscrub Description",
        "Website",
    ]
    columns_to_convert_to_strings = ["Name", "Top Level Category", "Secondary Category"]
    df[columns_to_remove_non_strings] = df[columns_to_remove_non_strings].map(
        lambda x: x if isinstance(x, str) else ""
    )
    df[columns_to_convert_to_strings] = df[columns_to_convert_to_strings].map(
        lambda x: str(x)
    )
    categories = ["Top Level Category", "Secondary Category"]
    df[categories] = df[categories].replace(to_replace=[None, ""], value="N/A")
    return df


def calculate_similarity(embedding1, embedding2):
    # Remove the eval() since embeddings are already numpy arrays
    embedding_1 = (
        embedding1 if isinstance(embedding1, np.ndarray) else np.array(embedding1)
    )
    embedding_2 = (
        embedding2 if isinstance(embedding2, np.ndarray) else np.array(embedding2)
    )

    # Ensure embeddings are 2D for cosine_similarity
    if embedding_1.ndim == 1:
        embedding_1 = embedding_1.reshape(1, -1)
    if embedding_2.ndim == 1:
        embedding_2 = embedding_2.reshape(1, -1)

    similarity = cosine_similarity(embedding_1, embedding_2)[0][0]
    return similarity


def data_pipeline(file_path, text_columns):
    df = ingest_data(file_path)
    print(df.info())
    df = preprocessing(df)
    if "Embeddings" not in df.columns:
        df = get_embeddings(df, text_columns)
    df.to_pickle("output.pkl")


if __name__ == "__main__":
    text_columns = ["Description", "Description.1", "Sourcscrub Description"]
    data_pipeline(
        file_path="../innovius_case_study_data.xlsx", text_columns=text_columns
    )
    df = pd.read_pickle("output.pkl")

    # Sample two random rows
    # best_sim = 0
    # best_desc_1 = ""
    # best_desc_2 = ""
    # for i in range(100000):
    #     random_rows = df.sample(n=2)
    #
    #     embeddings_1 = random_rows.iloc[0]["embeddings"]
    #     embeddings_2 = random_rows.iloc[1]["embeddings"]
    #
    #     sim = calculate_similarity(embeddings_1, embeddings_2)
    #     if best_sim < sim:
    #         best_sim = sim
    #         best_desc_1 = random_rows.iloc[0]
    #         best_desc_2 = random_rows.iloc[1]
    #
    # print("Similarity:", best_sim)
    # for text_col in text_columns:
    #     print(f"{text_col} 1: {best_desc_1[text_col]}")
    #     print(f"{text_col} 2: {best_desc_2[text_col]}")
