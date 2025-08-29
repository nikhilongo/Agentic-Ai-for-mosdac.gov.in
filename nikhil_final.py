import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFH42cxWbXA4EUkZ1qyu2v_6Lq0qoLzMw"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List

# ✅ Your components
from weather_tool import weather_guess  # @tool
from rag import retrieve      # @tool
from weather_too import format_answer
#from llm_setup import llm  # Your Gemini Flash 2 LLM

llm_with_weather = llm.bind_tools([weather_guess])

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_weather.invoke(state["messages"])]}

weather_tools_node = ToolNode([weather_guess])

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

def router_node(state: MessagesState) -> Dict[str, list]:
    return {"messages": state["messages"]}

def route_decision(state: MessagesState) -> str:
    user_query = [m.content for m in state["messages"] if m.type == "human"][-1]
    prompt = (
        "You are a router. Decide whether this query is about weather (temperature, forecast, etc.) "
        "or about static content like ISRO, satellites, or documents. "
        "Reply with only one word: 'weather' or 'rag'.\n\n"
        f"Query: {user_query}"
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    route = result.content.strip().lower()
    return "weather" if "weather" in route else "rag"

graph_builder = StateGraph(MessagesState)

# ✅ Add nodes
graph_builder.add_node("router", router_node)

# Weather branch
graph_builder.add_node("tool_calling_llm", tool_calling_llm)
graph_builder.add_node("tools", weather_tools_node)  # MUST be "tools" for tools_condition
graph_builder.add_node("format_answer", format_answer)

# RAG branch
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools_rag", rag_tools_node)
graph_builder.add_node("generate", generate)

# ✅ Entry point
graph_builder.set_entry_point("router")

# ✅ Routing based on user query
graph_builder.add_conditional_edges(
    "router",
    route_decision,  # decision logic returns "weather" or "rag"
    {
        "weather": "tool_calling_llm",
        "rag": "query_or_respond"
    }
)

# ✅ Weather branch flow
graph_builder.add_conditional_edges("tool_calling_llm", tools_condition)
graph_builder.add_edge("tools", "format_answer")
graph_builder.add_edge("format_answer", END)

# ✅ RAG branch flow (custom mapping: tools → tools_rag)
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

# ✅ Compile graph
graph = graph_builder.compile()

# query2 = [HumanMessage(content="what is insat?")]
# result = graph.invoke({"messages":query2})

# for m in result["messages"]:
#     m.pretty_print()

# query1 = [HumanMessage(content="What is the max temprature in Delhi for next 2 days? ")]
# result = graph.invoke({"messages":query1})

# for m in result["messages"]:
#     m.pretty_print()    