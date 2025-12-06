import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

class ScraperService:
    @staticmethod
    def scrape_page(url: str) -> List[Dict[str, str]]:
        """
        Fetches and parses data from a given webpage using BeautifulSoup.
        Extracts headings, links, and paragraphs.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        
        content = []
        for heading in soup.find_all(["h2", "h3"]):
            content.append({"heading": heading.get_text(strip=True)})
        for link in soup.find_all("a", href=True):
            content.append({
                "text": link.get_text(strip=True),
                "url": link["href"]
            })
        for para in soup.find_all("p"):
            text = para.get_text(strip=True)
            if text:
                content.append({"paragraph": text})
        return content
