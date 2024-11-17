"""
Company Comparison Tool
----------------------
A Streamlit application for comparing companies based on their descriptions using NLP embeddings.
The app allows users to:
1. Select and compare two companies
2. View similarity scores
3. Find similar companies
4. Analyze company details and categories

Dependencies:
    - streamlit
    - pandas
    - numpy
    - sentence-transformers
    - scikit-learn
    - plotly

Author: Brevinh Pham
Date: 2024
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Configuration ----
st.set_page_config(layout="wide", page_title="Company Comparison Tool")

# ---- Type Definitions ----
CompanyData = Dict[str, Union[str, float, np.ndarray]]
CompanyInfo = List[Dict[str, Union[str, int]]]
FilteredCompanies = List[Dict[str, Union[str, int]]]
DataFrameResult = Tuple[
    Optional[pd.DataFrame], Optional[CompanyInfo], Optional[List[str]]
]


# ---- Data Loading Functions ----
@st.cache_resource
def load_model() -> Optional[SentenceTransformer]:
    """
    Load the sentence transformer model with caching.

    Returns:
        Optional[SentenceTransformer]: Loaded model or None if loading fails
    """
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_data() -> DataFrameResult:
    """
    Load and preprocess company data from pickle files.
    Attempts to combine part1.pkl and part2.pkl if output.pkl doesn't exist.

    Returns:
        Tuple containing:
        - DataFrame or None: Processed company data
        - List[Dict] or None: Company info for dropdowns
        - List[str] or None: Category list
    """
    try:
        # Combine split files if necessary
        if os.path.exists("part1.pkl") and os.path.exists("part2.pkl"):
            part1 = pd.read_pickle("part1.pkl")
            part2 = pd.read_pickle("part2.pkl")
            df = pd.concat([part1, part2], axis=0)
            df.to_pickle("output.pkl")
            st.success(
                "Successfully combined 'part1.pkl' and 'part2.pkl' into 'output.pkl'."
            )
        else:
            st.error("Required data files not found!")
            return None, None, None

        # Validate required columns
        required_columns = [
            "Organization Id",
            "Name",
            "Top Level Category",
            "embeddings",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None, None, None

        # Process data
        df["Top Level Category"] = df["Top Level Category"].fillna("Uncategorized")
        company_info = df[["Organization Id", "Name", "Top Level Category"]].to_dict(
            "records"
        )
        categories = ["All"] + sorted(df["Top Level Category"].unique().tolist())

        return df, company_info, categories

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


# ---- Similarity Calculation Functions ----
@st.cache_data
def get_company_similarity_cache(company1_id: int, company2_id: int) -> float:
    """
    Calculate similarity between two companies using cached embeddings.

    Args:
        company1_id: First company's ID
        company2_id: Second company's ID

    Returns:
        float: Similarity score between 0 and 1
    """
    df = load_data()[0]
    company1_data = df[df["Organization Id"] == company1_id].iloc[0]
    company2_data = df[df["Organization Id"] == company2_id].iloc[0]

    embedding_1 = company1_data["embeddings"]
    embedding_2 = company2_data["embeddings"]

    # Reshape embeddings if necessary
    embedding_1 = embedding_1.reshape(1, -1) if embedding_1.ndim == 1 else embedding_1
    embedding_2 = embedding_2.reshape(1, -1) if embedding_2.ndim == 1 else embedding_2

    return float(cosine_similarity(embedding_1, embedding_2)[0][0])


@st.cache_data
def get_similar_companies_cache(company_id: int, top_n: int = 5) -> List[Dict]:
    """
    Find the top N most similar companies to a given company.

    Args:
        company_id: Target company's ID
        top_n: Number of similar companies to return

    Returns:
        List[Dict]: List of similar companies with their details
    """
    df = load_data()[0]
    company = df[df["Organization Id"] == company_id].iloc[0]
    company_embedding = company["embeddings"].reshape(1, -1)

    # Vectorized similarity calculation
    all_embeddings = np.vstack(df["embeddings"].values)
    similarities = cosine_similarity(company_embedding, all_embeddings)[0]

    # Get top N similar companies (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]

    return [
        {
            "id": df.iloc[idx]["Organization Id"],
            "name": df.iloc[idx]["Name"],
            "category": df.iloc[idx].get("Top Level Category", "No Category Found"),
            "similarity": similarities[idx],
            "description": df.iloc[idx].get(
                "Description",
                df.iloc[idx].get(
                    "Sourcscrub Description",
                    df.iloc[idx].get("Description.1", "No Description Found"),
                ),
            ),
            "employee_count": df.iloc[idx].get("Employee Count", "N/A"),
            "website": df.iloc[idx].get("Website", "N/A"),
        }
        for idx in similar_indices
    ]


# ---- UI Helper Functions ----
def display_company_details(company_data: Dict) -> None:
    """
    Display company details in a consistent format.

    Args:
        company_data: Dictionary containing company information
    """
    category = company_data["Top Level Category"]
    if category == "nan":
        category = "N/A"
    st.write(f"**Category:** {category}")
    st.write(f"**Employees:** {company_data.get('Employee Count', 'N/A')}")
    st.write(f"**Website:** {company_data.get('Website', 'N/A')}")

    with st.expander("View Descriptions"):
        st.write("**Main Description:**")
        st.write(company_data.get("Description", "No description available"))
        st.write("**Sourcscrub Description:**")
        st.write(company_data.get("Sourcscrub Description", "No description available"))
        if "Description.1" in company_data:
            st.write("**Additional Description:**")
            st.write(company_data["Description.1"])


def create_similarity_gauge(similarity: float) -> go.Figure:
    """
    Create a gauge chart for displaying similarity scores.

    Args:
        similarity: Similarity score between 0 and 1

    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=similarity * 100,
            title={"text": "Similarity Score"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgray"},
                    {"range": [33, 66], "color": "gray"},
                    {"range": [66, 100], "color": "darkgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": similarity * 100,
                },
            },
        )
    ).update_layout(height=300)


def filter_companies(
    company_info: List[Dict], category: str, search_term: str
) -> FilteredCompanies:
    """
    Filter companies based on category and search term.

    Args:
        company_info: List of company information dictionaries
        category: Category to filter by
        search_term: Search term to filter company names

    Returns:
        FilteredCompanies: Filtered list of companies
    """
    filtered = company_info
    if category != "All":
        filtered = [c for c in filtered if c["Top Level Category"] == category]
    if search_term:
        filtered = [c for c in filtered if search_term.lower() in c["Name"].lower()]
    return filtered


def main():
    st.title("Company Description Comparison")

    # Load data and model
    with st.spinner("Loading data and model..."):
        model = load_model()
        result = load_data()

        if result[0] is None:
            st.error("Failed to load data. Please check the errors above.")
            return

        df, company_info, categories = result

    # Sidebar debug information
    with st.sidebar:
        st.header("Debug Information")
        with st.expander("Debug Details"):
            st.write(f"Total companies: {len(df)}")
            st.write(f"Available columns: {df.columns.tolist()}")
            st.write(
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Company 1")

        # Independent filters for Company 1
        selected_category_1 = st.selectbox("Filter by Category", categories, key="cat1")
        search_term_1 = st.text_input("Search companies", "", key="search1")

        # Filter companies for Company 1
        filtered_companies_1 = filter_companies(
            company_info, selected_category_1, search_term_1
        )

        company1 = st.selectbox(
            "Select first company",
            options=[c["Organization Id"] for c in filtered_companies_1],
            format_func=lambda x: next(
                c["Name"] for c in filtered_companies_1 if c["Organization Id"] == x
            ),
        )

        if company1:
            company1_data = df[df["Organization Id"] == company1].iloc[0]
            display_company_details(company1_data)

    # Company 2 selection and display
    with col2:
        st.subheader("Company 2")

        # Independent filters for Company 2
        selected_category_2 = st.selectbox("Filter by Category", categories, key="cat2")
        search_term_2 = st.text_input("Search companies", "", key="search2")

        # Filter companies for Company 2
        filtered_companies_2 = filter_companies(
            company_info, selected_category_2, search_term_2
        )

        company2 = st.selectbox(
            "Select second company",
            options=[c["Organization Id"] for c in filtered_companies_2],
            format_func=lambda x: next(
                c["Name"] for c in filtered_companies_2 if c["Organization Id"] == x
            ),
            key="company2",
        )

        if company2:
            company2_data = df[df["Organization Id"] == company2].iloc[0]
            display_company_details(company2_data)

    if company1 and company2:
        st.header("Comparison Analysis")

        # Calculate similarity
        similarity = get_company_similarity_cache(company1, company2)

        # Display results in columns
        col3, col4 = st.columns([1, 2])

        with col3:
            st.plotly_chart(
                create_similarity_gauge(similarity), use_container_width=True
            )

        with col4:
            # Add button for computing similar companies
            if st.button("Find Top 5 Similar Companies", key="find_similar"):
                with st.spinner("Computing similar companies..."):
                    similar_companies = get_similar_companies_cache(company1)

                    if similar_companies:
                        similar_df = pd.DataFrame(similar_companies)
                        fig = px.bar(
                            similar_df,
                            x="name",
                            y="similarity",
                            title=f"Top 5 Similar Companies to {company1_data['Name']}",
                            labels={
                                "name": "Company Name",
                                "similarity": "Similarity Score",
                            },
                            color="similarity",
                            color_continuous_scale="Blues",
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=300)
                        st.plotly_chart(fig, use_container_width=True)

                        # Store the results in session state for use in tabs
                        st.session_state.similar_companies = similar_companies
            else:
                st.info("Click the button to find similar companies")

        # Detailed analysis section
        st.header("Detailed Analysis")
        tabs = st.tabs(["Similar Companies", "Comparison Details", "Category Analysis"])

        with tabs[0]:
            if "similar_companies" in st.session_state:
                for company in st.session_state.similar_companies:
                    with st.expander(f"{company['name']} ({company['category']})"):
                        st.write(f"**Similarity Score:** {company['similarity']:.2%}")
                        st.write(f"**Employees:** {company['employee_count']}")
                        st.write(f"**Website:** {company['website']}")
                        st.write("**Description:**")
                        st.write(company["description"])
            else:
                st.info(
                    "Click the 'Find Top 5 Similar Companies' button above to see similar companies"
                )

        with tabs[1]:
            col5, col6 = st.columns(2)
            with col5:
                st.write("**Company 1 Details**")
                st.write(f"Name: {company1_data['Name']}")
                st.write(f"Category: {company1_data['Top Level Category']}")
                st.write(f"Employees: {company1_data.get('Employee Count', 'N/A')}")

            with col6:
                st.write("**Company 2 Details**")
                st.write(f"Name: {company2_data['Name']}")
                st.write(f"Category: {company2_data['Top Level Category']}")
                st.write(f"Employees: {company2_data.get('Employee Count', 'N/A')}")

        with tabs[2]:
            if (
                company1_data["Top Level Category"]
                == company2_data["Top Level Category"]
            ):
                st.success(
                    f"Both companies are in the same category: {company1_data['Top Level Category']}"
                )
            else:
                st.warning("Companies are in different categories")
                st.write(f"Company 1 Category: {company1_data['Top Level Category']}")
                st.write(f"Company 2 Category: {company2_data['Top Level Category']}")


if __name__ == "__main__":
    main()
