import dotenv
import os

dotenv.load_dotenv()

from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
import requests
import json

OPEN_API_KEY = os.getenv("OPEN_API_KEY", "")
# print(OPEN_API_KEY)


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int


geolocator = Nominatim(user_agent="weather-app")


class SearchInput(BaseModel):
    location: str = Field(description="The city and state, e.g., San Francisco")
    date: str = Field(
        description="the forecasting date for when to get the weather format (yyyy-mm-dd)"
    )


@tool("get_weather_forecast", args_schema=SearchInput, return_direct=True)
def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a list dictionary with the time and temperature for each hour."""
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}"
            )
            data = response.json()
            return {
                time: temp
                for time, temp in zip(
                    data["hourly"]["time"], data["hourly"]["temperature_2m"]
                )
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}


tools = [get_weather_forecast]
tools_by_name = {tool.name: tool for tool in tools}

# Create LLM class
llm = ChatOpenAI(
    api_key=OPEN_API_KEY,
    model="gpt-4o",  # replace with "gemini-2.0-flash"
    temperature=1.0,
    max_tokens=None,
)

# Bind tools to the model
model = llm.bind_tools([get_weather_forecast])

# Test the model with tools
response = model.invoke("What is the weather in Berlin on 12th of March 2025?")
print(response)


# this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
# system_prompt = SystemMessage(
#     "You are a helpful assistant that use tools to access and retrieve information from a weather API. Today is 2025-03-04. Help the user with their questions. Use the history to answer the question."
# )


# Define our tool node
def call_tool(state: AgentState):
    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool by name
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # Invoke the model with the system prompt and the messages
    response = model.invoke(state["messages"], config)
    # We return a list, because this will get added to the existing messages state using the add_messages reducer
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    # If the last message is not a tool call, then we finish
    if not messages[-1].tool_calls:
        return "end"
    # default to continue
    return "continue"


# Define a new graph with our state
workflow = StateGraph(AgentState)

# 1. Add our nodes
workflow.add_node("llm", call_model)
workflow.add_node("tools", call_tool)
# 2. Set the entrypoint as `agent`, this is the first node called
workflow.set_entry_point("llm")
# 3. Add a conditional edge after the `llm` node is called.
workflow.add_conditional_edges(
    # Edge is used after the `llm` node is called.
    "llm",
    # The function that will determine which node is called next.
    should_continue,
    # Mapping for where to go next, keys are strings from the function return, and the values are other nodes.
    # END is a special node marking that the graph is finish.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)
# 4. Add a normal edge after `tools` is called, `llm` node is called next.
workflow.add_edge("tools", "llm")

# Now we can compile and visualize our graph
graph = workflow.compile()

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# Create our initial message dictionary
inputs = {"messages": [("user", "How is the weather in Berlin on 12th of March 2025?")]}

# call our graph with streaming to see the steps

for state in graph.stream(inputs, stream_mode="values"):
    last_message = state["messages"][-1]
    last_message.pretty_print()

state["messages"].append(("user", "Would it be in Munich warmer?"))


for state in graph.stream(state, stream_mode="values"):
    last_message = state["messages"][-1]
    last_message.pretty_print()
