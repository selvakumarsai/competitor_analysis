from IPython import get_ipython
from IPython.display import display, Markdown
import os
import requests
import langgraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import SystemMessage, HumanMessage
from getpass import getpass

# Define the GraphState
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        location: The location provided by the user.
        radius_km: The search radius in kilometers.
        store_type: The type of store to search for.
        competitors: A list of competitor information.
        prompt: The formatted prompt for the LLM.
        report: The generated competitor analysis report.
    """
    location: str
    radius_km: str
    store_type: str
    competitors: Annotated[list, operator.add]
    prompt: str
    report: str

# Define the graph nodes
def collect_user_info(state: GraphState) -> GraphState:
    """Collects user input for location, radius, and store type."""
    location = input("Enter location: Werribee, Melbourne ")
    radius_km = input("Enter search radius in kilometers: 1 ")
    store_type = input("Enter store type: Clothing Store")
    return {
        "location": location,
        "radius_km": radius_km,
        "store_type": store_type
    }

def search_competitors(state: GraphState) -> GraphState:
    """Find nearby competitors using Google Places API."""
    location = state.get("location")
    radius_km = state.get("radius_km")
    store_type = state.get("store_type")

    if not all([location, radius_km, store_type]):
        raise ValueError("Missing required input for search_competitors")

    # Ensure you have GOOGLE_API_KEY defined, perhaps by getting it earlier in the script
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY'] # Assuming you set this as an environment variable
    except KeyError:
        GOOGLE_API_KEY = getpass('Enter Google API Key: ')

    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{store_type} near {location}",
        "radius": int(float(radius_km) * 1000),
        "key": GOOGLE_API_KEY
    }
    r = requests.get(endpoint, params=params)
    r.raise_for_status()
    return {"competitors": r.json().get("results", [])}

def format_prompt(state: GraphState) -> GraphState:
    """Build analysis prompt for LLM based on competitor data."""
    location = state.get("location")
    competitors = state.get("competitors", [])

    if not location or not competitors:
         return {"prompt": "Could not generate a detailed report due to missing competitor data."}

    summary = "\n".join([
        f"{c.get('name','N/A')} — {c.get('formatted_address','N/A')} | Rating: {c.get('rating','N/A')}"
        for c in competitors
    ])
    prompt_text = (
        f"You are an expert retail strategist. Analyze the following clothing stores in {location}:\n\n"
        f"{summary}\n\n"
        "Provide:\n"
        "- Estimated peak footfall hours\n"
        "- Competitive positioning strategy\n"
        "- Opportunities for new or existing stores"
    )
    return {"prompt": prompt_text}

def generate_report(state: GraphState) -> GraphState:
    """Generate strategic insights using the LLM."""
    prompt = state.get("prompt")

    if not prompt:
        return {"report": "No prompt available to generate report."}

    # Ensure you have OPENAI_API_KEY defined, perhaps by getting it earlier in the script
    try:
        OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] # Assuming you set this as an environment variable
    except KeyError:
         OPENAI_API_KEY = getpass('Enter Open AI API Key: ')

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)

    messages = [
        SystemMessage(content="You are a helpful business analyst."),
        HumanMessage(content=prompt)
    ]

    resp = llm.generate([messages])
    return {"report": resp.generations[0][0].text.strip()}

# Build and compile the graph
builder = StateGraph(GraphState)

builder.add_node("collect_user_info", collect_user_info)
builder.add_node("search_competitors", search_competitors)
builder.add_node("format_prompt", format_prompt)
builder.add_node("generate_report", generate_report)

builder.set_entry_point("collect_user_info")
builder.set_finish_point("generate_report")

builder.add_edge("collect_user_info", "search_competitors")
builder.add_edge("search_competitors", "format_prompt")
builder.add_edge("format_prompt", "generate_report")

graph = builder.compile()

# Execute the graph and display the report
# You will need to provide initial state if not collecting input interactively
# For a script, you might pre-define inputs or get them as command line arguments
# For demonstration, we'll run with interactive input as in the original notebook
inputs = {} # Initialize inputs, will be populated by collect_user_info

state = graph.invoke(inputs)

report = state.get("report") if isinstance(state, dict) else None

if report:
    display(Markdown("# 📊 Competitor Insights Report"))
    display(Markdown(report))
else:
    print("No report generated. Check inputs or API keys.")