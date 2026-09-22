import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega as variáveis de ambiente do arquivo .env (API Key)
load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(page_title="Chat com PDF - RAG", page_icon="📄")
st.title("📄 Chat com seus Documentos (RAG)")
st.write("Faça o upload de um PDF e tire dúvidas sobre o conteúdo usando a IA do Gemini.")

# Verificar se a chave da API foi configurada
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "cole_sua_chave_aqui":
    st.warning("⚠️ Por favor, adicione sua GOOGLE_API_KEY no arquivo .env e reinicie a aplicação.")
    st.stop()

# Configurar modelo LLM (Gemini) e Embeddings (HuggingFace, locais e gratuitos)
@st.cache_resource
def get_models():
    # Usando o Gemini 1.5 Flash (rápido e excelente para tarefas de RAG)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # Usando um modelo de embeddings leve para transformar o texto em vetores
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = get_models()

# Upload de arquivo na interface
uploaded_file = st.file_uploader("Envie seu arquivo PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processando o PDF... Isso pode levar alguns segundos."):
        # Salvar o arquivo temporariamente para a biblioteca PyPDFLoader conseguir ler
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 1. Carregar o texto do PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # 2. Quebrar o texto em Chunks (Pedaços menores)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Criar os Embeddings e salvar no banco de dados vetorial ChromaDB
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Criar o mecanismo de busca (Retriever)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Traz os 3 pedaços mais relevantes

        # 4. Configurar a cadeia de RAG (Prompt + Busca + LLM)
        system_prompt = (
            "Você é um assistente útil e educado, especialista em analisar documentos. "
            "Use OS SEGUINTES trechos de contexto recuperado do documento para responder à pergunta do usuário. "
            "Se a resposta não estiver no contexto, diga que não encontrou a informação no documento fornecido. "
            "Seja claro, objetivo e responda sempre em Português do Brasil.\n\n"
            "Contexto do documento:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        st.success("PDF processado e indexado com sucesso! Faça sua pergunta abaixo.")

    # --- Lógica de Interface do Chat (Streamlit) ---
    
    # Inicializar histórico do chat na sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico antigo na tela
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Receber nova pergunta do usuário
    if prompt_text := st.chat_input("Ex: Qual é o tema principal deste documento?"):
        
        # Exibir a pergunta imediatamente na tela
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Gerar e exibir a resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Procurando no documento e pensando..."):
                response = rag_chain.invoke({"input": prompt_text})
                answer = response["answer"]
                st.markdown(answer)
        
        # Salvar a resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": answer})
