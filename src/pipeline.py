import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer

from text_preprocessing_helpers import text_preprocessing_pipeline

# Initialize model and tokenizer
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
tqdm.pandas()


def ingest_data(file_path: str) -> pd.DataFrame:
    """
    Load company data from an Excel file.

    Args:
        file_path (str): Path to the Excel file containing company data.

    Returns:
        pd.DataFrame: Loaded DataFrame containing company information.
    """
    df = pd.read_excel(file_path)
    return df


def split_into_chunks(text: str, tokenizer, max_length: int = 512) -> list:
    """
    Split text into chunks that can be processed by the model.

    Args:
        text (str): Input text to be split into chunks.
        tokenizer: HuggingFace tokenizer instance.
        max_length (int, optional): Maximum length of each chunk in tokens. Defaults to 512.

    Returns:
        list: List of text chunks, each within the token limit.
    """
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]


def process_chunk(chunk_data: tuple) -> pd.DataFrame:
    """
    Process a chunk of data to generate embeddings for text columns.

    This function is designed to run in parallel, with each worker process having
    its own model instance.

    Args:
        chunk_data (tuple): Tuple containing (DataFrame chunk, text columns to process, batch size).

    Returns:
        pd.DataFrame: Processed chunk with added embeddings column.
    """
    chunk, text_columns, batch_size = chunk_data
    # Initialize the model inside the worker process
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
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings.extend(batch_embeddings)

    chunk["embeddings"] = embeddings
    return chunk


def calculate_embedding(text: str, tokenizer, model) -> np.ndarray:
    """
    Calculate embedding for a text string, handling long texts by chunking.

    Args:
        text (str): Input text to generate embedding for.
        tokenizer: HuggingFace tokenizer instance.
        model: SentenceTransformer model instance.

    Returns:
        np.ndarray: Mean embedding vector across all chunks of the input text.
    """
    tokens = tokenizer.tokenize(text)
    max_length = 512
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]

    chunk_embeddings = [
        model.encode(tokenizer.convert_tokens_to_string(chunk)) for chunk in chunks
    ]

    return np.mean(chunk_embeddings, axis=0)


def get_embeddings(
    df: pd.DataFrame, text_columns: list, n_workers: int = 4, batch_size: int = 32
) -> pd.DataFrame:
    """
    Generate embeddings for text columns in parallel using multiple worker processes.

    Args:
        df (pd.DataFrame): Input DataFrame containing text columns.
        text_columns (list): List of column names containing text to be embedded.
        n_workers (int, optional): Number of parallel worker processes. Defaults to 4.
        batch_size (int, optional): Batch size for processing texts. Defaults to 32.

    Returns:
        pd.DataFrame: DataFrame with added embeddings column.
    """
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


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by handling duplicates, non-string values, and missing categories.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with cleaned and standardized values.
    """
    df = df.drop_duplicates()
    columns_to_remove_non_strings = [
        "Description",
        "Description.1",
        "Sourcscrub Description",
        "Website",
    ]
    columns_to_convert_to_strings = ["Name", "Top Level Category", "Secondary Category"]

    # Convert non-string values to empty strings in text columns
    df[columns_to_remove_non_strings] = df[columns_to_remove_non_strings].map(
        lambda x: x if isinstance(x, str) else ""
    )

    # Convert specified columns to strings
    df[columns_to_convert_to_strings] = df[columns_to_convert_to_strings].map(
        lambda x: str(x)
    )

    # Replace missing categories with 'N/A'
    categories = ["Top Level Category", "Secondary Category"]
    df[categories] = df[categories].replace(to_replace=[None, ""], value="N/A")
    return df


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        embedding1 (np.ndarray): First embedding vector.
        embedding2 (np.ndarray): Second embedding vector.

    Returns:
        float: Cosine similarity score between the two embeddings.
    """
    # Convert inputs to numpy arrays if needed
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


def data_pipeline(file_path: str, text_columns: list) -> None:
    """
    Main data processing pipeline that loads data, preprocesses it, generates embeddings,
    and saves the result.

    Args:
        file_path (str): Path to the input Excel file.
        text_columns (list): List of column names containing text to be embedded.
    """
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
