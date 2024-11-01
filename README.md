# ALAN - Programming Logic and Learning Assistant 🤓 (telegram version)

ALAN is a programming logic and learning assistant developed at the Institute of Computing (IComp) of the Federal University of Amazonas (UFAM). This bot is designed to assist users with questions about programming, particularly in Python and C, using a Telegram chatbot interface.

## Features

- **Programming Assistance**: ALAN can provide guidance and answers to questions related to programming, specifically focused on Python and C.
- **Chatbot Integration**: The assistant works as a Telegram chatbot, allowing seamless interaction through chat commands and messages.
- **Experimental Technology**: This bot is based on LMStudio and LLaMA 3.2 models, making it an educational experiment to explore the use of large language models in assisting with code and computing concepts.

## How to Use

1. **Start the Bot**: Use the `/start` command to initiate a conversation with ALAN. The bot will introduce itself and provide you with some options.
2. **Ask Questions**: You can ask ALAN any programming-related question, especially those involving Python or C.
3. **Disclaimer**: Type `/disclaimer` to view the disclaimer regarding the use of the bot.

## Commands

- `/start`: Start a conversation with ALAN.
- `/disclaimer`: View the disclaimer about the bot.

## Disclaimer

This chatbot is an educational experiment created by Prof. Juan from the Institute of Computing (IComp) at UFAM. There is no guarantee regarding the accuracy or functionality of the code or responses provided by the bot. Please use the content generated carefully and verify its correctness.

## Configuration
1. **Telegram Bot Token**: Replace `"telegram token"` with your actual Telegram bot token.
   - To have your own chatbot, first you need to generate a bot using the Telegram BotFather.
2. **LMStudio API**: Ensure that LMStudio is running locally and accessible at `http://127.0.0.1:1234/v1/chat/completions`.

## Replication
To run ALAN locally, follow these steps:

1. **Clone the Repository**

2. **Install Dependencies:** 
```bash
   pip install requests python-telegram-bot
```

3. **Run the Bot:**
```bash
   python3 alan_bot.py
```

**Contributing**

Contributions are welcome! If you have suggestions or would like to add new features, feel free to open an issue or submit a pull request.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Acknowledgements**

LMStudio and LLaMA 3.2 for the backend AI capabilities.

The Institute of Computing (IComp) at UFAM for supporting this educational project.

**If the server is up and running, you can try this Telegram bot:** [chat with ALAN on Telegram](https://t.me/Alanzito_bot)

