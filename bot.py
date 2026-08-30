import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from langchain_core.messages import HumanMessage
from ai_agent import agent

load_dotenv()
TOKEN = os.getenv("TELEGRAM_KEY")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", 10000))

def extract_text(msg_content):
    if isinstance(msg_content, str):
        return msg_content
    elif isinstance(msg_content, list):
        texts = []
        for item in msg_content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
            elif hasattr(item, "text"):
                texts.append(item.text)
        return " ".join(texts)
    elif isinstance(msg_content, dict) and "text" in msg_content:
        return msg_content["text"]
    return str(msg_content)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    chat_id = str(update.effective_chat.id)

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": chat_id}}
        )

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

        final_reply = extract_text(result["messages"][-1].content)
        await update.message.reply_text(final_reply)
    except Exception as e:
        print(f"Agent Processing Error: {e}")
        await update.message.reply_text("Kuch error aa gaya hai, please thodi der baad dobara try karein.")

if __name__ == "__main__":
    print(f"Starting Telegram Webhook Engine on Port {PORT}...")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if RENDER_EXTERNAL_URL:
        webhook_full_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TOKEN}"
        print(f"Registering Webhook with Telegram: {webhook_full_url}")

        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TOKEN,
            webhook_url=webhook_full_url,
            drop_pending_updates=True
        )
    else:
        print("Running in Local Polling Mode...")
        app.run_polling(drop_pending_updates=True)