# Chat com Documentos (Sistema RAG) 📄🤖

Este é um projeto desenvolvido para demonstrar a arquitetura RAG (Retrieval-Augmented Generation) aplicada a documentos PDF. O sistema permite que o usuário faça upload de um PDF e converse com uma Inteligência Artificial sobre o conteúdo dele.

## 🛠️ Tecnologias Utilizadas

- **Python** como linguagem principal.
- **Streamlit** para o Frontend / Interface do Chat.
- **LangChain** para a orquestração do RAG e ferramentas de LLM.
- **Google Gemini 1.5 Flash** como o modelo de linguagem (LLM).
- **HuggingFace** (`all-MiniLM-L6-v2`) para geração de Embeddings gratuitos e locais.
- **ChromaDB** como Banco de Dados Vetorial.

## 🚀 Como rodar o projeto localmente

1. Clone este repositório.
2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/Scripts/activate # No Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Crie um arquivo `.env` na raiz do projeto e adicione a sua chave da API do Google:
   ```env
   GOOGLE_API_KEY=sua_chave_api_aqui
   ```
5. Inicie a aplicação web:
   ```bash
   streamlit run app.py
   ```
