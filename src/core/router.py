from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from src.core.llm import llm

def route_decision(state: MessagesState) -> str:
    """
    Decides whether the query is about weather or RAG.
    """
    # Get the last human message
    user_query = next((m.content for m in reversed(state["messages"]) if m.type == "human"), None)
    
    if not user_query:
        return "rag" # Default to RAG if no query found (shouldn't happen)

    prompt = (
        "You are a router. Decide whether this query is about weather (temperature, forecast, etc.) "
        "or about static content like ISRO, satellites, or documents. "
        "Reply with only one word: 'weather' or 'rag'.\n\n"
        f"Query: {user_query}"
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    route = result.content.strip().lower()
    return "weather" if "weather" in route else "rag"
