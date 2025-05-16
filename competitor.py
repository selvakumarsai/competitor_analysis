# streamlit_competitor_insights_app.py
# ====================================
# Streamlit application for AI-powered competitor insights for clothing stores

import os
import requests
import streamlit as st
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------
# Configuration: API Keys
# ---------------------------------

# Load API keys from Streamlit secrets or environment
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not OPENAI_API_KEY:
    st.error("🔑 OpenAI API key not found. Set OPENAI_API_KEY in Streamlit secrets or environment.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("🔑 Google API key not found. Set GOOGLE_API_KEY in Streamlit secrets or environment.")
    st.stop()

# Initialize the language model
llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# ---------------------------------
# Helper functions
# ---------------------------------

def search_competitors(location: str, radius_km: float, store_type: str) -> List[Dict]:
    """
    Query Google Places API for nearby competitors.
    """
    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{store_type} near {location}",
        "radius": int(radius_km * 1000),
        "key": GOOGLE_API_KEY
    }
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json().get("results", [])


def format_prompt(location: str, competitors: List[Dict]) -> str:
    """
    Create a prompt for the LLM based on competitor data.
    """
    if not competitors:
        return "No competitors data to analyze."
    lines = []
    for comp in competitors:
        name = comp.get("name", "N/A")
        addr = comp.get("formatted_address", "N/A")
        rating = comp.get("rating", "N/A")
        lines.append(f"{name} — {addr} | Rating: {rating}")
    summary = "\n".join(lines)
    prompt = (
        f"You are a retail market strategist. Analyze the following clothing stores in {location}:\n\n"
        f"{summary}\n\n"
        "Provide:\n"
        "- Estimated peak customer footfall hours\n"
        "- Competitive positioning strategies\n"
        "- Opportunities for new or existing stores\n"
    )
    return prompt


def generate_insights(prompt: str) -> str:
    """
    Use the LLM to generate a strategic insights report.
    """
    messages = [
        SystemMessage(content="You are a helpful business insights analyst."),
        HumanMessage(content=prompt)
    ]
    resp = llm.generate([messages])
    return resp.generations[0][0].text.strip()

# ---------------------------------
# Streamlit UI
# ---------------------------------

def main():
    st.set_page_config(page_title="Competitor Insights", layout="wide")
    st.title("📊 AI-Powered Competitor Insights for Clothing Retail")

    st.markdown(
        """
        This app helps business owners, marketing teams, analysts, and investors
        analyze nearby clothing store competitors by location, providing insights
        on peak hours, footfall trends, and strategic recommendations.
        """
    )

    # Sidebar inputs
    st.sidebar.header("🔍 Search Parameters")
    location = st.sidebar.text_input("Location (e.g., Fitzroy, Melbourne)", value="")
    radius = st.sidebar.number_input("Search radius (km)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
    store_type = st.sidebar.text_input("Store type", value="clothing store")
    run_query = st.sidebar.button("Generate Report")

    if not location:
        st.info("Please enter a location to begin analysis.")
        return

    if run_query:
        with st.spinner("Fetching competitors and generating report..."):
            try:
                competitors = search_competitors(location, radius, store_type)
                if not competitors:
                    st.warning("No competitors found. Try a wider radius or different location.")
                    return

                # Display competitor list
                st.subheader(f"🛍️ Competitors near {location} (within {radius} km)")
                for idx, comp in enumerate(competitors[:10], start=1):
                    st.write(
                        f"{idx}. **{comp.get('name','N/A')}** — "
                        f"{comp.get('formatted_address','N/A')} | Rating: {comp.get('rating','N/A')}"
                    )

                # Build prompt and generate report
                prompt = format_prompt(location, competitors)
                insights = generate_insights(prompt)

                # Display AI-generated insights
                st.subheader("🤖 AI-Generated Strategic Insights")
                st.markdown(insights)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
