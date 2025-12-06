from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List

from src.core.llm import llm
from src.core.router import route_decision
from src.tools.weather_tool import weather_guess
from src.tools.rag_tool import retrieve

# --- Weather Branch Components ---
llm_with_weather = llm.bind_tools([weather_guess])

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_weather.invoke(state["messages"])]}

weather_tools_node = ToolNode([weather_guess])

def format_answer(state: MessagesState):
    # This function formats the weather tool output
    # It assumes the last message is the tool output
    # We need to generate a final response based on the tool output
    
    # Simple pass-through for now, or we can use LLM to format
    # Let's use LLM to format the final answer based on tool output
    
    last_message = state["messages"][-1]
    if last_message.type == "tool":
         # Generate a friendly response using the tool output
        prompt = [
            SystemMessage(content="You are a helpful weather assistant. Use the provided weather data to answer the user's question clearly."),
            *state["messages"]
        ]
        response = llm.invoke(prompt)
        return {"messages": [response]}
    return {"messages": []}


# --- RAG Branch Components ---
llm_with_rag = llm.bind_tools([retrieve])

def query_or_respond(state: MessagesState):
    return {"messages": [llm_with_rag.invoke(state["messages"])]}

rag_tools_node = ToolNode([retrieve])

def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    return {"messages": [llm.invoke(prompt)]}

# --- Graph Construction ---

def router_node(state: MessagesState) -> Dict[str, list]:
    return {"messages": state["messages"]}

graph_builder = StateGraph(MessagesState)

# Add nodes
graph_builder.add_node("router", router_node)

# Weather branch
graph_builder.add_node("tool_calling_llm", tool_calling_llm)
graph_builder.add_node("tools", weather_tools_node)
graph_builder.add_node("format_answer", format_answer)

# RAG branch
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools_rag", rag_tools_node)
graph_builder.add_node("generate", generate)

# Entry point
graph_builder.set_entry_point("router")

# Routing
graph_builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "weather": "tool_calling_llm",
        "rag": "query_or_respond"
    }
)

# Weather branch flow
graph_builder.add_conditional_edges("tool_calling_llm", tools_condition)
graph_builder.add_edge("tools", "format_answer")
graph_builder.add_edge("format_answer", END)

# RAG branch flow
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {
        "tools": "tools_rag",
        "default": END
    }
)
graph_builder.add_edge("tools_rag", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()
