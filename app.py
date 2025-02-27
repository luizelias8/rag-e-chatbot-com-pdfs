import os
import tempfile
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Carrega as variáveis de ambiente
load_dotenv()

# Inicializar o modelo de chat
chat = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'), # Chave de API
    model='gpt-4o' # Modelo LLM a ser usado
)

def obter_base_vetores_dos_pdfs(arquivos):
    """Carrega o conteúdo de múltiplos arquivos PDF usando LangChain, divide o texto em pedaços e cria uma base vetorial."""

    # Lista para armazenar todos os documentos carregados
    documentos = []

    # Cria arquivos temporários para cada PDF enviado
    for arquivo in arquivos:
        # Cria um arquivo temporário para salvar o conteúdo do arquivo enviado
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as arquivo_temporario:
            arquivo_temporario.write(arquivo.getvalue())
            caminho_arquivo = arquivo_temporario.name # Obtém o caminho do arquivo temporário

        # Usa o PyPDFLoader do LangChain para carregar o PDF
        carregador = PyPDFLoader(caminho_arquivo)
        # Carrega o documento e adiciona à lista de documentos
        documentos.extend(carregador.load())

        # Remove o arquivo temporário após o carregamento
        os.unlink(caminho_arquivo)

    # Configura o divisor de texto em pedaços
    divisor_texto = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ' ', ''], # Define o separador como uma quebra de linha
        chunk_size=500, # Define o tamanho de cada pedaço de texto
        chunk_overlap=200, # Define a sobreposição entre os pedaços
    )

    # Divide o texto do documento em pedaços
    documentos_divididos = divisor_texto.split_documents(documentos)

    # Configura o modelo de embeddings para gerar representações vetoriais
    modelo_embeddings = OpenAIEmbeddings()

    # Cria uma base vetorial persistente usando os textos em pedaços
    base_vetores = FAISS.from_documents(documentos_divididos, modelo_embeddings)
    return base_vetores

def montar_prompt(fragmentos, pergunta):
    """Monta manualmente o prompt com os fragmentos e o histórico de conversa."""

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
                    st.session_state.historico_chat.append(AIMessage(content='Olá, me faça perguntas a respeito do conteúdo carregado'))
                    # Processa o PDF e gera a base vetorial
                    st.session_state.base_vetores = obter_base_vetores_dos_pdfs(arquivos_pdfs)

                # Mostra mensagem de sucesso após o processamento
                st.success('Documentos processados com sucesso!')

    if st.session_state.base_vetores is not None:
        # Exibe o histórico do chat na interface
        for mensagem in st.session_state.historico_chat:
            if isinstance(mensagem, AIMessage): # Mensagem do chatbot
                with st.chat_message('ai'):
                    st.write(mensagem.content)
            elif isinstance(mensagem, HumanMessage): # Mensagem do usuário
                with st.chat_message('human'):
                    st.write(mensagem.content)

        # Captura a entrada do usuário no chat
        pergunta = st.chat_input('Digite sua mensagem aqui...')

        # Processa a mensagem do usuário e gera resposta
        if pergunta is not None and pergunta != '':
            # Exibir a pergunta do usuário no chat
            with st.chat_message('human'):
                st.write(pergunta)

            # Cria uma mensagem no chat para o assistente
            with st.chat_message('ai'):
                placeholder = st.empty()

                # Exibe uma mensagem temporária no chat enquanto os documentos relevantes são recuperados com base na pergunta do usuário.
                placeholder.write('Recuperando...')

                # Recuperar documentos relevantes com base na pergunta usando o banco vetorial
                documentos_relevantes = st.session_state.base_vetores.max_marginal_relevance_search(pergunta, k=3, fetch_k=10)

                # Exibe uma mensagem temporária no chat indicando que o modelo está processando a resposta com base nos fragmentos recuperados.
                placeholder.write('Gerando resposta...')

                # Montar o prompt com os fragmentos
                prompt = montar_prompt(documentos_relevantes, pergunta)

                # Adiciona o prompt com os trechos e a pergunta ao histórico
                st.session_state.historico_chat.append(HumanMessage(content=prompt))

                # Obtém resposta do modelo considerando o histórico
                resposta = chat.stream(st.session_state.historico_chat)

                # Inicializa uma string para armazenar a resposta completa
                resposta_completa = ''

                # Acumula as partes da resposta
                for parte in placeholder.write_stream(resposta):
                    resposta_completa += parte # Acumula o texto gerado

            # Limpa o histórico antes de adicionar a resposta, removendo o prompt montado
            st.session_state.historico_chat.pop() # Remove a última, que seria o prompt montado
            st.session_state.historico_chat.append(HumanMessage(content=pergunta)) # Adiciona apenas a pergunta ao histórico
            st.session_state.historico_chat.append(AIMessage(content=resposta_completa)) # Adicionar a resposta do modelo ao histórico de mensagens

# Executa a aplicação se o script for chamado diretamente
if __name__ == '__main__':
    main()
