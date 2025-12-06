# MOSDAC Chatbot Agent

A modular, intelligent chatbot designed to provide weather information and answer queries about ISRO/MOSDAC using RAG (Retrieval-Augmented Generation). Built with LangGraph, LangChain, and Streamlit.

## 🚀 Features

*   **Weather Forecast**: Fetches real-time 7-day weather forecasts for Indian cities using the MOSDAC API.
*   **Knowledge Base (RAG)**: Answers questions about ISRO, satellites, and technical documents by retrieving information from a local vector database.
*   **Intelligent Routing**: Automatically decides whether to use the Weather tool or the RAG tool based on the user's query.
*   **Modular Architecture**: Clean, scalable code structure separated into core, services, tools, and config.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Orchestration**: LangGraph, LangChain
*   **LLM**: Google Gemini (via `langchain-google-genai`)
*   **Vector Store**: ChromaDB
*   **Data Processing**: BeautifulSoup4 (for scraping), RecursiveCharacterTextSplitter

## 📂 Project Structure

```
/src
  /config       # Configuration and environment variables
  /core         # Core logic (LLM, Graph, Router, Vector Store)
  /services     # External services (Weather API, Scraper)
  /tools        # LangChain tools
  /utils        # Utility functions (Data loading)
  /data         # Data files (JSON, etc.)
  app.py        # Main Streamlit application
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nikhilongo/Agentic-Ai-for-mosdac.gov.in.git
    cd Agentic-Ai-for-mosdac.gov.in
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have `streamlit`, `langchain`, `langgraph`, `langchain-google-genai`, `chromadb`, `beautifulsoup4`, `geopy`, `python-dotenv` installed)*

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## 🏃‍♂️ Usage

Run the application using the provided batch script (Windows) or directly via Streamlit:

**Option 1: Batch Script**
Double-click `run_app.bat` or run:
```powershell
.\run_app.bat
```

**Option 2: Manual Command**
```bash
streamlit run src/app.py
```

## 🧠 How it Works

1.  **Input**: User enters a query in the Streamlit UI.
2.  **Routing**: The `Router` node analyzes the intent (Weather vs. General Info).
3.  **Execution**:
    *   **Weather**: Calls `WeatherService` -> MOSDAC API -> Formats response.
    *   **RAG**: Embeds query -> Searches `ChromaDB` -> Generates answer using retrieved context.
4.  **Response**: The final answer is displayed in the chat interface.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
