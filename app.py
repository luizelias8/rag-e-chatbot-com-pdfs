import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Carrega as variáveis de ambiente
load_dotenv()

# Inicializar o modelo de chat
chat = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'), # Chave de API
    model='gpt-4-1106-preview', # Modelo LLM a ser usado
    temperature=0.2, # Baixa temperatura para respostas mais precisas
    max_tokens=500 # Limite de tokens na resposta
)

def obter_base_vetores_dos_pdfs(arquivos):
    """Carrega o conteúdo de múltiplos arquivos PDF, divide o texto em pedaços e cria uma base vetorial."""

    # Variável para armazenar o texto extraído de todos os PDFs
    documento = ''

    # Itera sobre cada arquivo enviado
    for arquivo in arquivos:
        # Cria um leitor para o arquivo PDF
        leitor_pdf = PdfReader(arquivo)
        # Itera sobre cada página do PDF
        for pagina in leitor_pdf.pages:
            # Adiciona o texto extraído à variável
            documento += pagina.extract_text()

    # Configura o divisor de texto em pedaços
    divisor_texto = CharacterTextSplitter(
        separator='\n', # Define o separador como uma quebra de linha
        chunk_size=500, # Define o tamanho de cada pedaço de texto
        chunk_overlap=200, # Define a sobreposição entre os pedaços
        length_function=len # Usa o comprimento do texto para controle de tamanho
    )

    # Divide o texto do documento em pedaços
    pedacos_documento = divisor_texto.split_text(documento)

    # Configura o modelo de embeddings para gerar representações vetoriais
    modelo_embeddings = OpenAIEmbeddings()

    # Cria uma base vetorial persistente usando os textos em pedaços
    base_vetores = FAISS.from_texts(pedacos_documento, modelo_embeddings)
    return base_vetores

def verificar_se_eh_resumo(pergunta):
    """Verifica se a pergunta é uma solicitação de resumo."""
    palavras_chave_resumo = ['resumo', 'resuma', 'sintetiza', 'sumarize']
    return any(palavra in pergunta.lower() for palavra in palavras_chave_resumo)

def obter_documentos_relevantes(base_vetores, pergunta, eh_pedido_resumo=False):
    """Obtém documentos relevantes da base vetorial, com quantidade adaptativa."""
    # Se for um pedido de resumo, recupera mais documentos
    k = 10 if eh_pedido_resumo else 3
    return base_vetores.similarity_search(pergunta, k=k)

def montar_prompt(fragmentos, pergunta, eh_pedido_resumo=False):
    """Monta manualmente o prompt com os fragmentos e o histórico de conversa."""

    if eh_pedido_resumo:
        template = """
        Crie um resumo conciso e bem estruturado usando os trechos do documento fornecidos abaixo.
        Organize o resumo de forma coerente, conectando as ideias principais encontradas nos diversos trechos.
        Se perceber que há partes faltando ou desconexas, indique isso no resumo.

        ### Trechos:
        {fragmentos}

        ### Solicitação:
        {pergunta}
        """
    else:
        template = """
        Use os trechos fornecidos para responder à pergunta do usuário de forma clara e concisa.
        Se necessário, complemente a resposta utilizando o histórico do chat.
        Se não souber a resposta com base nos trechos fornecidos e no histórico do chat, diga que não sabe, sem tentar adivinhar ou inventar informações.
        Se possível, seja direto e objetivo ao responder.

        ### Trechos:
        {fragmentos}

        ### Pergunta:
        {pergunta}
        """

    # Juntar todos os fragmentos em um único texto
    contexto = '\n'.join([f'{indice}. {fragmento.page_content}\n' for indice, fragmento in enumerate(fragmentos,1)])

    # Criar e formatar o prompt
    prompt = template.format(fragmentos=contexto, pergunta=pergunta)

    return prompt

def main():
    """Função principal para configurar e executar a interface da aplicação Streamlit."""
    # Inicializa o histórico de chat na sessão, se ainda não existir
    if 'historico_chat' not in st.session_state:
        st.session_state.historico_chat = []
    # Inicializa a base de vetores na sessão, se ainda não existir
    if 'base_vetores' not in st.session_state:
        st.session_state.base_vetores = None
    # Inicializa o estado de desabilitado do prompt se não existir
    if 'prompt_sistema_desabilitado' not in st.session_state:
        st.session_state.prompt_sistema_desabilitado = False

    # Configura o título e o ícone da página
    st.set_page_config(page_title='Chat com arquivos PDF', page_icon='🤖')
    st.title('Chat com arquivos PDF')

    # Configura a barra lateral para upload de arquivos e criação de persona
    with st.sidebar:
        # Adicionar um campo para criar um prompt de sistema
        st.header('🎭 Persona do Chatbot')
        prompt_sistema = st.text_area(
            'Defina o comportamento do bot aqui:',
            placeholder='Ex.: Você é um assistente especializado em análise de dados financeiros.',
            help='Insira um prompt para personalizar a persona do chatbot.',
            disabled=st.session_state.prompt_sistema_desabilitado
        )

        # Cabeçalho das configurações
        st.header('📁 Upload de Documentos')
        # Permite envio de arquivos PDF
        arquivos_pdfs = st.file_uploader(
            'Selecione os arquivos PDF',
            type='pdf',
            accept_multiple_files=True,
            help='Você pode fazer upload de múltiplos arquivos PDF'
        )

        # Se os arquivos foram enviados e o botão foi pressionado
        if arquivos_pdfs:
            if st.button('Processar PDFs', use_container_width=True):
                # Mostra spinner durante processamento
                with st.spinner('Processando documentos...'):
                    # Adiciona o prompt do sistema primeiro se existir e não tiver sido adicionado ainda
                    if prompt_sistema and not any(isinstance(m, SystemMessage) for m in st.session_state.historico_chat):
                        st.session_state.historico_chat.insert(0, SystemMessage(content=prompt_sistema))
                        # Desabilita o prompt após processar
                        st.session_state.prompt_sistema_desabilitado = True

                    # Inicializa o histórico de chat com a primeira mensagem do bot
                    st.session_state.historico_chat.append(AIMessage(content='Olá, sou um bot. Como posso ajudar?'))
                    # Processa o PDF e gera a base vetorial
                    st.session_state.base_vetores = obter_base_vetores_dos_pdfs(arquivos_pdfs)

                # Mostra mensagem de sucesso após o processamento
                st.success('Documentos processados com sucesso!')

    if st.session_state.base_vetores is not None:
        # Captura a entrada do usuário no chat
        pergunta = st.chat_input('Digite sua mensagem aqui...')

        # Processa a mensagem do usuário e gera resposta
        if pergunta is not None and pergunta != '':
            # Verifica se é pedido de resumo
            eh_pedido_resumo = verificar_se_eh_resumo(pergunta)

            # Recuperar documentos relevantes com base na pergunta
            documentos_relevantes = obter_documentos_relevantes(
                st.session_state.base_vetores,
                pergunta,
                eh_pedido_resumo
            )

            # Montar o prompt com os fragmentos
            prompt = montar_prompt(documentos_relevantes, pergunta, eh_pedido_resumo)

            # Adiciona o prompt com os trechos e a pergunta ao histórico
            st.session_state.historico_chat.append(HumanMessage(content=prompt))

            # Exibir um spinner enquanto o modelo gera a resposta
            with st.spinner('Gerando resposta...'):
                resposta = chat.invoke(st.session_state.historico_chat) # Obter a resposta do modelo

            # Limpa o histórico antes de adicionar a resposta, removendo o prompt montado
            st.session_state.historico_chat.pop() # Remove a última, que seria o prompt montado
            st.session_state.historico_chat.append(HumanMessage(content=pergunta)) # Adiciona apenas a pergunta ao histórico
            st.session_state.historico_chat.append(AIMessage(content=resposta.content)) # Adicionar a resposta do modelo ao histórico de mensagens

        # Exibe o histórico do chat na interface
        for mensagem in st.session_state.historico_chat:
            if isinstance(mensagem, AIMessage): # Mensagem do chatbot
                with st.chat_message('ai'):
                    st.write(mensagem.content)
            elif isinstance(mensagem, HumanMessage): # Mensagem do usuário
                with st.chat_message('human'):
                    st.write(mensagem.content)

# Executa a aplicação se o script for chamado diretamente
if __name__ == '__main__':
    main()
