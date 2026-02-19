import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load models and data
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    ranking_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(ranking_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        ranking_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return embedder, tokenizer, model, device

embedder, ranking_tokenizer, ranking_model, device = load_models()

# Load FAISS index and product sample
products_df = pd.read_csv("data/products_50k.csv")
index = faiss.read_index("data/faiss_index_50k.bin")

# Retrieval Agent (FAISS)
def retrieve_products(query_text, top_k=5):
    query_vec = embedder.encode([query_text], convert_to_numpy=True)
    query_vec = query_vec.astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)
    retrieved = products_df.iloc[indices[0]]
    return retrieved

# Ranking Agent (LLM re-ranker)
def ranking_agent(query_text, retrieved_df):
    # Build product descriptions for LLM
    candidate_descriptions = ""
    for i, row in enumerate(retrieved_df.itertuples()):
        candidate_descriptions += (
            f"[{i}] Title: {row.title}\n"
            f"Description: {row.description}\n\n"
        )

    prompt = f"""
You are a product ranking agent. Rank the following products by how well they match the user's query.

User query: "{query_text}"

Products:
{candidate_descriptions}

Return ONLY a Python list of indices in best-to-worst order, like: [2, 0, 1, 3, 4]
"""

    inputs = ranking_tokenizer(prompt, return_tensors="pt").to(device)

    output = ranking_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

    response = ranking_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract list from output
    import re
    match = re.search(r"\[(.*?)\]", response)
    if match:
        order = [int(x.strip()) for x in match.group(1).split(",")]
    else:
        order = list(range(len(retrieved_df)))

    ranked_df = retrieved_df.iloc[order]
    return ranked_df, response

# Pricing Agent
def pricing_agent(product_row):
    title = product_row.title
    desc = product_row.description
    price = product_row.get("price", None)
    segment = product_row.get("price_segment", None)

    prompt = f"""
You are a pricing analysis agent. Analyze whether the product is fairly priced for typical consumers.

Product Title: {title}
Description: {desc}
Listed Price: {price}
Price Segment: {segment}

Provide a JSON object with keys:
- fairness
- expected_rating
- recommended_range
- explanation
    """

    inputs = ranking_tokenizer(prompt, return_tensors="pt").to(device)

    output = ranking_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    response = ranking_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Orchestrator Agent
def orchestrator_agent(user_query, top_k=5):
    retrieved = retrieve_products(user_query, top_k)
    ranked, ranking_output = ranking_agent(user_query, retrieved)
    top_product = ranked.iloc[0]
    pricing_output = pricing_agent(top_product)

    return {
        "query": user_query,
        "recommended_product": top_product,
        "ranking_output": ranking_output,
        "pricing_output": pricing_output
    }

# Streamlit Web UI
st.title("E-Commerce Multi-Agent Recommendation System")
st.write("Enter a natural-language shopping request to get a personalized recommendation.")

# Text input
user_query = st.text_input("What are you looking for?")

# Run button
if st.button("Recommend"):
    if not user_query.strip():
        st.warning("Please enter a query before running the recommendation.")
    else:
        with st.spinner("Running multi-agent pipeline..."):
            result = orchestrator_agent(user_query)

        # Display recommended product
        st.subheader("Top Recommendation")
        product = result["recommended_product"]

        st.write(f"**Item ID:** {product['item_id']}")
        st.write(f"**Title:** {product['title']}")
        st.write(f"**Description:** {product['description']}")
        st.write(f"**Price:** {product.get('price', 'N/A')}  ({product.get('price_segment', 'N/A')})")

        # Ranking explanation
        st.subheader("Ranking Agent Output")
        st.code(result["ranking_output"])

        # Pricing explanation
        st.subheader("Pricing Agent Output")
        st.code(result["pricing_output"])
