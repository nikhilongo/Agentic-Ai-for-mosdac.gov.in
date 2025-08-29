import json
from langchain_core.documents import Document

def load_filtered_docs_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for page in data:
        content_parts = []

        # Title
        title = page.get("title", "")
        if title:
            content_parts.append(f"Title: {title}")

        tags = page.get("tags", {})

        # h1
        for h in tags.get("h1", []):
            content_parts.append(f"H1: {h}")

        # p
        for p in tags.get("p", []):
            if len(p.strip()) > 20:
                content_parts.append(p.strip())

        # table
        for table_text in tags.get("table", []):
            if len(table_text.strip()) > 30:
                content_parts.append("Table:\n" + table_text.strip())

        # meta (only description, abstract, keywords)
        for meta in tags.get("meta", []):
            if meta.get("name") in ["description", "abstract", "keywords"]:
                content_parts.append(f"{meta['name'].capitalize()}: {meta['content']}")

        # Combine all into a single text blob
        full_text = "\n\n".join(content_parts).strip()
        if full_text:
            documents.append(Document(page_content=full_text, metadata={"source": page["url"]}))

    return documents
