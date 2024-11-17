import os
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure page to use wide mode by default
st.set_page_config(layout="wide", page_title="Company Comparison Tool")


# Load model and data
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_data():
    try:
        # Check if 'output.pkl' exists; if not, attempt to combine parts
        if os.path.exists("part1.pkl") and os.path.exists("part2.pkl"):

            # Load part1.pkl and part2.pkl
            part1 = pd.read_pickle("part1.pkl")
            part2 = pd.read_pickle("part2.pkl")

            # Combine the DataFrames
            df = pd.concat([part1, part2], axis=0)

            # Save combined DataFrame as 'output.pkl' for future use
            df.to_pickle("output.pkl")
            st.success(
                "Successfully combined 'part1.pkl' and 'part2.pkl' into 'output.pkl'."
            )
        else:
            st.error(
                "Data file 'output.pkl' not found, and parts 'part1.pkl'/'part2.pkl' are missing!"
            )
            return None, None, None

        # Check if required columns exist
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

        # Handle missing values in Top Level Category
        df["Top Level Category"] = df["Top Level Category"].fillna("Uncategorized")

        company_info = df[["Organization Id", "Name", "Top Level Category"]].to_dict(
            "records"
        )
        categories = ["All"] + sorted(df["Top Level Category"].unique().tolist())

        return df, company_info, categories

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


# Cache similarity calculations
@st.cache_data
def get_company_similarity_cache(company1_id, company2_id):
    df = load_data()[0]
    company1_data = df[df["Organization Id"] == company1_id].iloc[0]
    company2_data = df[df["Organization Id"] == company2_id].iloc[0]

    embedding_1 = company1_data["embeddings"]
    embedding_2 = company2_data["embeddings"]

    if embedding_1.ndim == 1:
        embedding_1 = embedding_1.reshape(1, -1)
    if embedding_2.ndim == 1:
        embedding_2 = embedding_2.reshape(1, -1)

    return float(cosine_similarity(embedding_1, embedding_2)[0][0])


# Cache similar companies calculation
@st.cache_data
def get_similar_companies_cache(company_id, top_n=5):
    df = load_data()[0]
    company = df[df["Organization Id"] == company_id].iloc[0]
    company_embedding = company["embeddings"].reshape(1, -1)

    # Vectorized similarity calculation
    all_embeddings = np.vstack(df["embeddings"].values)
    similarities = cosine_similarity(company_embedding, all_embeddings)[0]

    # Get top N similar companies (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]

    similar_companies = []
    for idx in similar_indices:
        row = df.iloc[idx]
        similar_companies.append(
            {
                "id": row["Organization Id"],
                "name": row["Name"],
                "category": row.get("Top Level Category", row.get("No Category Found")),
                "similarity": similarities[idx],
                "description": row.get(
                    "Description",
                    row.get(
                        "Sourcscrub Description",
                        row.get("Description.1", "No Description Found"),
                    ),
                ),
                "employee_count": row.get("Employee Count", "N/A"),
                "website": row.get("Website", "N/A"),
            }
        )

    return similar_companies


def display_company_details(company_data):
    """Helper function to display company details consistently"""
    st.write(f"**Category:** {company_data['Top Level Category']}")
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


def create_similarity_gauge(similarity):
    """Helper function to create similarity gauge"""
    fig = go.Figure(
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
    )
    fig.update_layout(height=300)
    return fig


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

    # Sidebar filters and search
    with st.sidebar:
        st.header("Filters and Search")

        # Category filter
        selected_category = st.selectbox("Filter by Category", categories)

        # Search box
        search_term = st.text_input("Search companies", "")

        # Debug information in expander
        with st.expander("Debug Information"):
            st.write(f"Total companies: {len(df)}")
            st.write(f"Available columns: {df.columns.tolist()}")
            st.write(
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

    # Filter companies based on category and search
    filtered_companies = company_info
    if selected_category != "All":
        filtered_companies = [
            c
            for c in filtered_companies
            if c["Top Level Category"] == selected_category
        ]
    if search_term:
        filtered_companies = [
            c for c in filtered_companies if search_term.lower() in c["Name"].lower()
        ]

    # Main content area
    col1, col2 = st.columns(2)

    # Company selection and display
    with col1:
        st.subheader("Company 1")
        company1 = st.selectbox(
            "Select first company",
            options=[c["Organization Id"] for c in filtered_companies],
            format_func=lambda x: next(
                c["Name"] for c in filtered_companies if c["Organization Id"] == x
            ),
        )

        if company1:
            company1_data = df[df["Organization Id"] == company1].iloc[0]
            display_company_details(company1_data)

    with col2:
        st.subheader("Company 2")
        company2 = st.selectbox(
            "Select second company",
            options=[c["Organization Id"] for c in filtered_companies],
            format_func=lambda x: next(
                c["Name"] for c in filtered_companies if c["Organization Id"] == x
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
