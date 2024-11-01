import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# Create the application instance with the Telegram bot token
app = ApplicationBuilder() \
    .token("telegram token") \
    .connect_timeout(30) \
    .read_timeout(30) \
    .build()

# LMStudio API URL
LMSTUDIO_API_URL = 'http://127.0.0.1:1234/v1/chat/completions'

disclaimer_message = ('Assistente de programação desenvolvido pelo Prof. Juan do Instituto de Computação (IComp) da UFAM.\n\n'
                      'Disclaimer: este projeto é um experimento educacional e não há garantias sobre a precisão ou funcionamento dos códigos ou respostas fornecidas.' 
                      'Por favor, use com cautela e verifique sempre o conteúdo gerado.\n'
                      'https://github.com/juancolonna/ALAN/tree/main\n')

# Function to get response from LMStudio
async def get_lmstudio_response(prompt):  
    if len(prompt.split()) > 200:
        return 'O comprimento do texto execede a capacidade deste bot. Faça uma pergunta menor.\n'
    
    payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "user", "content": f"{prompt}"}
        ],
        "temperature": 0.3,
        "max_tokens": -1,
        "stream": False
    }
    try:
        response = requests.post(LMSTUDIO_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("message", {}).get("content", "Desculpe, não consegui gerar uma resposta.")
        else:
            return "Desculpe, não consegui gerar uma resposta."
    except requests.exceptions.RequestException as e:
        return "Erro: Não foi possível conectar ao LMStudio."

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Olá! Eu sou ALAN 🤓, o Assistente de Lógica e Aprendizado em Programação do Instituto de Computação (IComp) da UFAM.'
                                    'Sou baseado em tecnologias como LMStudio e LLama 3.2.\n' 
                                    'Pergunte-me qualquer coisa sobre programação ou computação, mas especialmente em Python ou C!\n\n'
                                    
                                    '/disclaimer')

# Start command handler
async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(disclaimer_message)

# Message handler to process user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    lmstudio_response = await get_lmstudio_response(user_message)
    await update.message.reply_text(lmstudio_response)


# Register handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("disclaimer", disclaimer))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
# Start the Bot
app.run_polling()