import streamlit as st
import re
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.settings import Settings
import wikipedia
import time as t

# Gemini
import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

load_dotenv()

os.environ["NLTK_DATA"] = os.path.expanduser("~/.nltk_data")

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    st.error("⚠️ GENAI_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

try:
    llm = Gemini(model="models/gemini-1.5-flash", api_key=api_key)
    embed_model = GeminiEmbedding(model="models/embedding-001", api_key=api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model

except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.stop()

def extract_url_and_query(message):
    """Extract URL and query from the message."""
    url_pattern = r"https?://[^\s]+"
    url_match = re.search(url_pattern, message)
    url = url_match.group(0) if url_match else None
    query = re.sub(url_pattern, "", message).strip()
    return query, url

def is_wikipedia_url(url):
    """Check if a URL is a valid Wikipedia URL."""
    if not url:
        return False
    return bool(re.match(r"https?://(www\.)?wikipedia\.org/wiki/", url) or 
                re.match(r"https?://[a-z]+\.wikipedia\.org/wiki/", url))

def extract_wikipedia_title(url):
    """Extract the title/topic from a Wikipedia URL."""
    if not url:
        return None
    
    match = re.search(r"/wiki/([^/]+)$", url)
    if match:
        title = match.group(1).replace("_", " ")
        import urllib.parse
        title = urllib.parse.unquote(title)
        return title
    
    return None

def fetch_wikipedia_data(query):
    """Fetch Wikipedia data based on the user query.
    If the first search result fails, try the next available page."""
    page_title = query  # Initialize page_title with query
    try:
        search_results = wikipedia.search(query)

        if not search_results:
            st.warning(f"No Wikipedia pages found for: {query}")
            return [], page_title

        for page_title in search_results:
            try:
                wiki_page = wikipedia.page(page_title, auto_suggest=False)

                docs = WikipediaReader().load_data(pages=[wiki_page.title])

                if docs:
                    st.success(f"{wiki_page.title} is loaded.")
                    return docs, page_title

            except wikipedia.DisambiguationError as e:
                st.warning(f"Disambiguation Error: {e.options}. Trying the next option...")
                continue 
            
            except wikipedia.PageError:
                st.warning(f"Page not found: {page_title}. Trying the next option...")
                continue

        st.error(f"Could not retrieve any valid Wikipedia pages for: {query}")
        return [], page_title

    except Exception as e:
        st.error(f"Error fetching Wikipedia data: {str(e)}")
        return [], page_title
    
def generate_response(index, cleaned_query, topic="General"):
    try:
        if index:
            QA_PROMPT_TMPL = """You are a helpful assistant.
            You will be given a query to answer, and a context from Wikipedia to help you answer the query.
            
            Current Wikipedia Topic: {topic}
            
            Context: {context_str}
            
            Query: {query_str}
            
            Try to use the provided context to answer the query, and do not try to guess if you don't have the needed information.
            Always be helpful and informative."""
            
            QA_PROMPT = PromptTemplate(template=QA_PROMPT_TMPL)

            query_engine = index.as_query_engine(
                text_qa_template=QA_PROMPT
            )
            
            response = query_engine.query(cleaned_query)
            return response.response if hasattr(response, "response") else str(response)
        else:
            return "Sorry, the knowledge base isn't available. Please provide a Wikipedia URL to load a topic."
    except Exception as e:
        return f"An error occurred: {str(e)}"

st.set_page_config(page_title="ChatWiki", layout="wide")
st.title("📚 ChatWiki")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "topic" not in st.session_state:
    st.session_state.topic = "Artificial Intelligence"

if "index" not in st.session_state:
    st.session_state.index = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask me anything about the current topic or paste a Wikipedia URL!")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        cleaned_query, url = extract_url_and_query(query)
        loading_new_kb = False
        response_text = ""

        if url and is_wikipedia_url(url):
            wiki_title = extract_wikipedia_title(url)
            if wiki_title:
                message_placeholder.markdown(f"Loading Wikipedia article: '{wiki_title}'...")
                loading_new_kb = True
                
                wiki_docs, new_wiki_title = fetch_wikipedia_data(wiki_title)
                if wiki_docs:
                    message_placeholder.markdown(f"Indexing article: '{new_wiki_title}'...")
                    st.session_state.index = VectorStoreIndex.from_documents(wiki_docs)
                    st.session_state.topic = wiki_title
                    
                    if cleaned_query:
                        response_text = generate_response(st.session_state.index, cleaned_query, wiki_title)
                    else:
                        response_text = f"I've loaded information about '{wiki_title}'. What would you like to know?"
                else:
                    response_text = f"I couldn't load the Wikipedia article for '{wiki_title}'. The topic may not exist or there was an error fetching the data."
        
        elif not loading_new_kb:
            if st.session_state.index is None:
                response_text = "No knowledge base loaded yet. Please provide a Wikipedia URL to get started."
            else:
                response_text = generate_response(st.session_state.index, cleaned_query, st.session_state.topic)

        message_placeholder.markdown(response_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})