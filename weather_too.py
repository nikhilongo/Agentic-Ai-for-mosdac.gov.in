import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFH42cxWbXA4EUkZ1qyu2v_6Lq0qoLzMw"

from langchain_google_genai import ChatGoogleGenerativeAI
from weather_tool import weather_guess

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
llm_with_tools = llm.bind_tools([weather_guess])

# weather_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from weather_tool import weather_guess
from typing import List, Dict

# Node to call LLM (decide whether to answer or call tool)
def tool_calling_llm(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def format_answer(state: Dict[str, List]):
    messages = state["messages"]
    tool_result = messages[-1].content  # tool output
    user_question = [m.content for m in messages if isinstance(m, HumanMessage)][-1]

    prompt = (
        f"The user asked: '{user_question}'\n\n"
        f"The tool returned this data:\n{tool_result}\n\n"
        "Please extract only the relevant information in a clear and friendly format."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# Build graph
builder = StateGraph(MessagesState)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([weather_guess]))
builder.add_node("format_answer", format_answer)

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "format_answer")
builder.add_edge("format_answer", END)

graph = builder.compile()
