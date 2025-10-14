import requests
from bs4 import BeautifulSoup
import json

def scrape_page(url):
    """
    Fetches and parses data from a given webpage using BeautifulSoup.
    Extracts headings, links, and paragraphs.
    """
    response = requests.get(url)
    response.raise_for_status()
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


def main():
    """
    Scrapes data from multiple MOSDAC pages and saves it to a JSON file.
    """
    urls = [
        "https://www.mosdac.gov.in/",
        "https://www.mosdac.gov.in/data",
        "https://www.mosdac.gov.in/inventory"
    ]

    all_data = {}
    for url in urls:
        print(f"Scraping: {url}")
        all_data[url] = scrape_page(url)

    with open("mosdac_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print("✅ Data from MOSDAC pages saved to 'mosdac_data.json'")


if __name__ == "__main__":
    main()
