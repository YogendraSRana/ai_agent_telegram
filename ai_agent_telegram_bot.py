import os
import requests
import subprocess
import base64
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

load_dotenv()

# 1. API Keys Setup
gemini_key = os.getenv("GOOGLE_API_KEY")
telegram_key = os.getenv("TELEGRAM_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# 2. Hybrid LLM Initialization (Notebook match)
gemini = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    api_key=gemini_key
)

# 3. Dynamic Paths Setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DB_DIR = BASE_DIR / "agent_db"
SHARED_FOLDER = DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 4. Local Vector DB / RAG Pipeline Setup (Ollama nomic-embed-text)
print(f"Checking data folder: {DATA_DIR}")
embedder = OllamaEmbeddings(model="nomic-embed-text")

sqlite_file = DB_DIR / "chroma.sqlite3"
if sqlite_file.exists():
    print(f"Found existing Chroma DB. Loading Vector Store from: {DB_DIR}")
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedder
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    all_texts = []
    print("Reading documents from Data folder to build agent_db...")
    
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() == ".pdf":
                try:
                    reader = PdfReader(str(file_path))
                    text = "".join([p.extract_text() or "" for p in reader.pages])
                    if text.strip():
                        all_texts.append(text)
                        print(f"Loaded PDF: {file_path.name}")
                except Exception as e:
                    print(f"Error reading PDF {file_path.name}: {e}")
            elif file_path.suffix.lower() == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():
                            all_texts.append(text)
                            print(f"Loaded TXT: {file_path.name}")
                except Exception as e:
                    print(f"Error reading TXT {file_path.name}: {e}")

    if not all_texts:
        print("No documents found. Using default system text to initialize database.")
        all_texts = ["Initial AI Assistant System Document."]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for t in all_texts:
        chunks.extend(splitter.split_text(t))

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedder,
        persist_directory=str(DB_DIR)
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"SUCCESS: agent_db created at: {DB_DIR}")

# 5. Helper function for Windows Notification
def _esc(s: str) -> str:
    return s.replace("'", "''")

# 6. Define Tools (Notebook & File Sharing Tools)
@tool
def get_marks(subject: str) -> str:
    """Use this function to get the marks for any subject."""
    return f"{subject}: 85/100"

@tool
def show_notification(title: str, message: str) -> str:
    """Show a desktop toast notification on Windows to alert the user."""
    ps = f'''
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null
    $t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(1)
    $t.GetElementsByTagName("text")[0].AppendChild($t.CreateTextNode('{_esc(title)}')) > $null
    $t.GetElementsByTagName("text")[1].AppendChild($t.CreateTextNode('{_esc(message)}')) > $null
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("AI Agent").Show($t)
    '''
    enc = base64.b64encode(ps.encode("utf-16-le")).decode()
    subprocess.run(["powershell", "-EncodedCommand", enc], capture_output=True)
    return f"Notification shown: {title}"

@tool
def get_news(topic: str) -> str:
    """Get the latest news headlines about any topic."""
    url = "https://newsapi.org/v2/everything"
    params = {"q": topic, "sortBy": "publishedAt", "pageSize": 5, "language": "en", "apiKey": news_api_key}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return f"News API error: {r.status_code}"
    articles = r.json().get("articles", [])
    if not articles:
        return f"No news found about {topic}"
    return "\n".join([f"- {a['title']} ({a['source']['name']})" for a in articles])

@tool
def rag_search_documents(question: str) -> str:
    """Search user's documents and notes to answer a question."""
    if not retriever:
        return "No documents uploaded in data folder."
    docs = retriever.invoke(question)
    return "\n\n".join(d.page_content for d in docs) if docs else "Nothing found in documents."

@tool
def list_files() -> str:
    """List all files available in the shared data folder."""
    files = [f.name for f in SHARED_FOLDER.iterdir() if f.is_file()]
    return "\n".join(files) if files else "Folder is empty."

@tool
def send_file(filename: str) -> str:
    """Send a file from shared folder. Filename must match list_files output."""
    path = (SHARED_FOLDER / filename).resolve()
    if SHARED_FOLDER.resolve() not in path.parents or not path.is_file():
        return f"No valid file named {filename}."
    return f"FILE:{path}"

@tool
def my_age()-> str:
    """Return the age of the user."""
    return "I am 25 years old."


# 7. Create Final Agent
tools = [my_age]
agent = create_agent(
    model=gemini,
    tools=tools,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(model=gemini, trigger=("messages", 20), keep=("messages", 6))
    ],
    system_prompt="""You are a smart AI Assistant on Telegram.
    - search_documents: for questions about user notes/data
    - send_file: when user asks to download or send a file (call list_files first if unsure)
    - get_news: for recent news/events
    - get_marks: for student marks
    - show_notification: to alert on the host desktop
    Keep responses concise and direct."""
)

def extract_text(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    parts = [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip() or "(no reply)"

def force_takeover():
    """Telegram conflict error hatane ke liye purane webhooks ko clear karta hai."""
    try:
        base = f"https://api.telegram.org/bot{telegram_key}"
        requests.post(f"{base}/deleteWebhook", params={"drop_pending_updates": True}, timeout=5)
    except Exception:
        pass

# 8. Telegram Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_text)]},
        config={"configurable": {"thread_id": chat_id}}
    )

    # Check for FILE: marker in intermediate tool calls
    for m in result["messages"]:
        content = m.content if isinstance(m.content, str) else ""
        if content.startswith("FILE:"):
            file_path = content[5:]
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                await update.message.reply_document(
                    document=file_bytes,
                    filename=os.path.basename(file_path)
                )

    final_reply = extract_text(result["messages"][-1])
    await update.message.reply_text(final_reply)

# 9. Run Bot
if __name__ == "__main__":
    force_takeover()
    print("Starting Telegram Bot...")
    
    app = (
        Application.builder()
        .token(telegram_key)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(poll_interval=2.0, timeout=30)