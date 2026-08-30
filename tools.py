import os
import pytz
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()
news_api_key = os.getenv("NEWS_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
SHARED_FOLDER = DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

@tool
def get_current_datetime() -> str:
    """Returns the current live date, day of week, year, and exact time in Indian Standard Time (IST)."""
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    return now.strftime("%A, %d %B %Y, %I:%M:%S %p (IST)")

@tool
def get_news(topic: str) -> str:
    """Get instant live news and headlines for any topic or query."""
    # 1. Primary: Fast Google News RSS
    try:
        url = f"https://news.google.com/rss/search?q={topic}&hl=en-IN&gl=IN&ceid=IN:en"
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            root = ET.fromstring(res.content)
            items = root.findall(".//item")[:4]
            if items:
                headlines = []
                for item in items:
                    t = item.find("title").text if item.find("title") is not None else ""
                    headlines.append(f"- {t}")
                return "\n".join(headlines)
    except Exception:
        pass

    # 2. Fallback: NewsAPI
    if news_api_key:
        try:
            api_url = "https://newsapi.org/v2/everything"
            params = {"q": topic, "sortBy": "publishedAt", "pageSize": 3, "language": "en", "apiKey": news_api_key}
            r = requests.get(api_url, params=params, timeout=4)
            if r.status_code == 200:
                arts = r.json().get("articles", [])
                if arts:
                    return "\n".join([f"- {a['title']}" for a in arts])
        except Exception:
            pass

    return f"No recent headlines found for '{topic}'."

@tool
def calculate(expression: str) -> str:
    """Safely evaluates basic mathematical calculations. Example: '45 * 12 + 100'"""
    allowed = set("0123456789+-*/(). %")
    if not all(c in allowed for c in expression):
        return "Invalid characters in mathematical expression."
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def list_files() -> str:
    """List all available documents in the shared data folder."""
    files = [f.name for f in SHARED_FOLDER.iterdir() if f.is_file()]
    return "\n".join(files) if files else "No documents in Data folder."

@tool
def send_file(filename: str) -> str:
    """Retrieve and send a file from shared folder."""
    path = (SHARED_FOLDER / filename).resolve()
    if SHARED_FOLDER.resolve() not in path.parents or not path.is_file():
        return f"No valid file named '{filename}' found."
    return f"FILE:{path}"