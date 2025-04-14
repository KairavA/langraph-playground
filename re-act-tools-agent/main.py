import os
from typing import Annotated, List, Sequence, TypedDict, Union, Dict, Any
import operator
from enum import Enum
import json
from datetime import datetime
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool, ToolException

from dotenv import load_dotenv

load_dotenv()
# Configuration
OPENAI_API_KEY = os.getenv("OPEN_API_KEY", "")
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0


# Define the agent state
class AgentState(TypedDict):
    messages: List[Union[AIMessage, HumanMessage]]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    next_step: str


# Define the routing enum
class NextStep(str, Enum):
    THINK = "think"
    USE_TOOLS = "use_tools"
    RESPOND = "respond"


# Tool definitions
@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query."""
    # In a real implementation, this would use a search API
    try:
        # Simulated response - in a real scenario would call an actual search API
        return f"Search results for '{query}': Found information about {query} including latest updates as of {datetime.now().strftime('%Y-%m-%d')}."
    except Exception as e:
        raise ToolException(f"Error searching web: {str(e)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Be cautious with eval - in production, use a safer approach or a specialized math library
        allowed_names = {"abs": abs, "round": round}
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        raise ToolException(f"Error calculating: {str(e)}")


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulate weather data - in a real implementation, would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temperatures = range(0, 35)
    import random

    condition = random.choice(weather_conditions)
    temp = random.choice(temperatures)
    return f"Current weather in {location}: {condition} with a temperature of {temp}°C."


@tool
def get_current_date() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Create a custom tool executor that doesn't rely on langgraph.prebuilt.ToolExecutor
class ToolExecutor:
    def __init__(self, tools: List[Any]):
        self.tools = {tool.name: tool for tool in tools}

    def invoke(self, tool_invocation: Dict[str, Any]) -> Any:
        """Invoke a tool with the given inputs."""
        tool_name = tool_invocation["name"]
        tool_input = tool_invocation["arguments"]

        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")

        tool = self.tools[tool_name]
        if isinstance(tool_input, dict):
            return tool(**tool_input)
        else:
            return tool(tool_input)


# Create the tools and tool executor
tools = [search_web, calculator, get_current_weather, get_current_date]
tool_executor = ToolExecutor(tools)

# Create system messages
ROUTER_SYSTEM_MESSAGE = """You are an advanced AI assistant that can think step-by-step or use tools when needed.

When faced with a user query, you should:
1. Analyze whether the query requires external information or can be answered through reasoning
2. If external information is needed, use the appropriate tools
3. If reasoning is sufficient, think step-by-step (chain-of-thought) to arrive at the answer
4. Provide a clear, helpful response to the user based on your analysis or tool results

Available tools:
{tool_descriptions}

Based on the conversation history, decide what to do next. Format your response as JSON:
{{
  "next_step": "think | use_tools | respond",
  "reasoning": "your reasoning here"
}}
"""

THINKING_SYSTEM_MESSAGE = """You are an advanced AI assistant that can think step-by-step.

Think step-by-step to solve this problem. Label your thinking as 'Chain of Thought:' followed by your reasoning.
"""

TOOL_SYSTEM_MESSAGE = """You are an advanced AI assistant that can use tools when needed.

Available tools:
{tool_descriptions}

What tool(s) should you use to answer this query? Format your response as JSON:
{{
  "name": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
"""

RESPONSE_SYSTEM_MESSAGE = """You are a helpful assistant. Provide a clear, concise response to the user's query based on your reasoning or tool results.

Synthesize all information and provide a final response to the user.
"""

tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

# Create the LLM
llm = ChatOpenAI(temperature=TEMPERATURE, model=MODEL_NAME, api_key=OPENAI_API_KEY)


# Define the agent functions
def route(state: AgentState) -> Dict[str, str]:
    """Determine the next step in the agent's process."""
    messages = state["messages"]

    # Create a system message with the tool descriptions
    system_message = SystemMessage(
        content=ROUTER_SYSTEM_MESSAGE.format(tool_descriptions=tool_descriptions)
    )

    # Combine with the conversation history
    full_messages = [system_message] + messages

    # Call the LLM to decide what to do next
    response = llm.invoke(full_messages)

    try:
        # Try to parse the response as JSON
        content = response.content
        parsed = json.loads(content)
        next_step = parsed.get("next_step", "think")
        if next_step not in [e.value for e in NextStep]:
            next_step = "think"  # Default to thinking if invalid step
    except:
        # If JSON parsing fails, default to thinking
        next_step = "think"

    return {"next_step": next_step}


def think(state: AgentState) -> AgentState:
    """Apply chain-of-thought reasoning to the problem."""
    messages = state["messages"]

    # Create a system message
    system_message = SystemMessage(content=THINKING_SYSTEM_MESSAGE)

    # Combine with the conversation history
    full_messages = [system_message] + messages

    # Generate a chain-of-thought analysis
    thinking_response = llm.invoke(full_messages)

    # Add the thinking to the messages
    new_messages = messages + [AIMessage(content=thinking_response.content)]

    # Update state
    return {"messages": new_messages, "next_step": "respond"}


def use_tools(state: AgentState) -> AgentState:
    """Use tools to gather information for the response."""
    messages = state["messages"]

    # Create a system message with the tool descriptions
    system_message = SystemMessage(
        content=TOOL_SYSTEM_MESSAGE.format(tool_descriptions=tool_descriptions)
    )

    # Combine with the conversation history
    full_messages = [system_message] + messages

    # Determine which tools to use
    tool_response = llm.invoke(full_messages)
    content = tool_response.content

    # Try to parse tool calls
    tool_calls = []
    try:
        if content.strip().startswith("{") and content.strip().endswith("}"):
            # Single tool call in JSON format
            parsed = json.loads(content)
            tool_calls.append(parsed)
        elif content.strip().startswith("[") and content.strip().endswith("]"):
            # Multiple tool calls in JSON array
            tool_calls = json.loads(content)
        else:
            # Try to extract JSON from the text
            import re

            json_matches = re.findall(r"\{.*?\}", content, re.DOTALL)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if "name" in parsed and "parameters" in parsed:
                        tool_calls.append(parsed)
                except:
                    continue
    except Exception as e:
        # If tool parsing fails, add error message
        error_msg = f"Error parsing tool calls: {str(e)}"
        new_messages = messages + [AIMessage(content=error_msg)]
        return {"messages": new_messages, "next_step": "respond"}

    # Execute the tools
    tool_results = []
    for call in tool_calls:
        if isinstance(call, dict) and "name" in call and "parameters" in call:
            try:
                tool_name = call["name"]
                parameters = call["parameters"]

                # Execute the tool
                result = tool_executor.invoke(
                    {"name": tool_name, "arguments": parameters}
                )
                tool_results.append(
                    {"tool": tool_name, "parameters": parameters, "result": result}
                )
            except Exception as e:
                tool_results.append(
                    {
                        "tool": call.get("name", "unknown"),
                        "parameters": call.get("parameters", {}),
                        "error": str(e),
                    }
                )

    # Add tool results to messages
    tool_messages = []
    for result in tool_results:
        if "error" in result:
            content = f"Tool '{result['tool']}' failed: {result['error']}"
        else:
            content = f"Tool '{result['tool']}' returned: {result['result']}"
        tool_messages.append(AIMessage(content=content))

    new_messages = messages + tool_messages

    # Update state
    return {
        "messages": new_messages,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "next_step": "respond",
    }


def respond(state: AgentState) -> AgentState:
    """Generate the final response to the user."""
    messages = state["messages"]

    # Create a system message
    system_message = SystemMessage(content=RESPONSE_SYSTEM_MESSAGE)

    # Combine with the conversation history
    full_messages = [system_message] + messages

    # Generate a response based on all accumulated information
    final_response = llm.invoke(full_messages)

    # Add the final response to messages
    new_messages = messages + [AIMessage(content=final_response.content)]

    # Update state and mark as complete
    return {"messages": new_messages, "next_step": END}


# Create the graph
def create_agent():
    # Initialize the workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route", route)
    workflow.add_node("think", think)
    workflow.add_node("use_tools", use_tools)
    workflow.add_node("respond", respond)

    # Add edges
    workflow.add_conditional_edges(
        "route",
        lambda x: x["next_step"],
        {"think": "think", "use_tools": "use_tools", "respond": "respond"},
    )
    workflow.add_edge("think", "respond")
    workflow.add_edge("use_tools", "respond")

    # Set the entry point
    workflow.set_entry_point("route")

    # Compile the graph
    return workflow.compile()


class LLMAgent:
    def __init__(self):
        self.graph = create_agent()
        self.state = {
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "next_step": "route",
        }

    def reset(self):
        """Reset the agent's state."""
        self.state = {
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "next_step": "route",
        }

    def process_message(self, message: str) -> str:
        """Process a user message and return the agent's response."""
        # Add the user message to the state
        self.state["messages"].append(HumanMessage(content=message))

        # Run the graph
        result = self.graph.invoke(self.state)

        # Update the state
        self.state = result

        # Return the last AI message
        last_message = self.state["messages"][-1]
        return last_message.content


# Example usage
if __name__ == "__main__":
    agent = LLMAgent()

    # Example queries to test the agent
    queries = [
        "What's 24 * 17?",
        "Can you explain quantum computing to me?",
        "What's the weather like in Paris?",
        "What are the key components of a good machine learning model?",
        "What's the current date?",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = agent.process_message(query)
        print(f"Agent: {response}")
        # Reset for next example
        agent.reset()
