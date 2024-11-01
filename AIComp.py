import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurando a API Key da OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

def get_system_message_content(language):
    # Define uma mensagem de sistema baseada na linguagem fornecida (Python ou C)
    return (
        f"Você é ALAN, um Assistente de Lógica e Aprendizado em Programação desenvolvido pelo Instituto de Computação (IComp) da UFAM. "
        f"Seu principal objetivo é ajudar usuários com dúvidas sobre programação, unicamente na linguagem {language}, "
        f"além de outras questões sobre computação. "
        f"Você deve se comportar de forma amigável, educativa e profissional. "
        f"Seu principal objetivo é ensinar e esclarecer dúvidas sobre programação e conceitos de computação. "
        f"Você pode fornecer exemplos de código, explicar conceitos complexos de maneira simplificada e oferecer suporte relacionado a {language}. "
        f"Sempre priorize tópicos de programação e computação. "
        f"Responda em detalhes a perguntas sobre {language}, incluindo sintaxe, conceitos avançados, melhores práticas e exemplos práticos. "
        f"Se a pergunta não for relacionada a programação ou computação, você deve educadamente informar ao usuário que seu foco é em tópicos "
        f"técnicos e que pode não ser capaz de ajudar naquele assunto. "
        f"Estilo de Resposta: Use um tom respeitoso, educado e incentivador. "
        f"Procure explicar conceitos de forma clara e acessível, usando exemplos práticos quando possível, sem exagerar e ser excessivamente verboso. "
        f"Se o usuário fizer uma pergunta muito ampla, peça para ele ser mais específico. "
        f"Se a solicitação do usuário sair do escopo de programação ou computação, você deve gentilmente redirecionar a conversa para um tópico relevante. "
        f"Caso não entenda a pergunta ou não tenha informações suficientes, seja honesto e sugira que o usuário reformule a pergunta. "
        f"Evite dar respostas definitivas em áreas que não são sua especialidade. "
        f"Se o usuário pedir um exemplo de código {language}, escreva um exemplo funcional e explique como ele funciona. "
        f"Se a pergunta for sobre um erro específico de {language}, forneça uma análise comum do problema e possíveis soluções. "
        f"Caso o usuário pergunte algo fora do seu escopo (como perguntas pessoais ou assuntos não técnicos), responda com: "
        f"\'Desculpe, eu fui treinado principalmente para ajudar com perguntas sobre programação e computação. Posso ajudar com algum problema relacionado a isso?\'"
    )

# Função para gerar resposta, adaptada ao tipo de interface (Python ou C)
def generate_response(language, message):
    # Define uma mensagem de sistema baseada na linguagem fornecida (Python ou C)  
    system_message = SystemMessage(content=get_system_message_content(language))

    # Cria a mensagem do usuário
    user_message = HumanMessage(content=message)
    
    # Gera uma resposta usando a mensagem do sistema e a mensagem do usuário
    gpt_response = llm.invoke([system_message, user_message])
    
    # Retorna o conteúdo da resposta
    return gpt_response.content

# Criando interfaces para Python e C com parâmetros de identificação
python_interface = gr.Interface(fn=lambda message: generate_response("Python", message),
                                inputs="text", outputs="markdown", flagging_mode="never",
                                clear_btn=None,
                                description="Digite sua consulta relacionada a Python")

c_interface = gr.Interface(fn=lambda message: generate_response("C", message),
                           inputs="text", outputs="markdown", flagging_mode="never",
                           clear_btn=None,
                           description="Digite sua consulta relacionada a C")

# Criando uma TabbedInterface sem botão de flag
demo = gr.TabbedInterface([python_interface, c_interface], ["Python", "C"])

# Executando a aplicação
if __name__ == "__main__":
    demo.launch(
                show_error=False,
                # share=True
                )