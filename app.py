import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import requests
import traceback
import re
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Set the page configuration
st.set_page_config(
    page_title="LangChain Groq",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set the title and description of the app
st.title("🦙LangChain: Summarize Youtube Videos and Web Pages")

st.write(
    "This app uses LangChain and Groq to summarize Youtube videos and web pages. "

)
st.subheader("Please enter a Youtube video URL or a web page URL to get started.")

# Get groq API key and summarization type from the user
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    st.subheader("Advanced Settings")
    summarization_type = st.radio(
        "Select Summarization Method",
        ["Basic (Stuff)", "Advanced (Map-Reduce)"],
        index=0
    )
    model_name = st.selectbox(
        "Select Groq Model",
        ["gemma2-9b-it", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

URL = st.text_input("Enter a Youtube video URL or a web page URL", label_visibility="collapsed")

# Define prompt templates
map_prompt_template = """Write a concise summary of the following content:
{text}
CONCISE SUMMARY:"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """Combine these summaries into a single comprehensive summary of about 300 words:
{text}
COMPREHENSIVE SUMMARY:"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

# For the direct "stuff" approach
stuff_prompt_template = """Summarize the following content in about 300-400 words:
Content: {text}
SUMMARY:"""
stuff_prompt = PromptTemplate(template=stuff_prompt_template, input_variables=["text"])

# Helper function to check if a URL is accessible
def is_url_accessible(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.head(url, headers=headers, timeout=5)
        return 200 <= response.status_code < 400
    except Exception:
        return False

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    # Regular YouTube URL
    youtube_regex = r'(https?://)?(www\.)?youtube\.(com|nl)/watch\?v=([^&]+)'
    match = re.match(youtube_regex, url)
    if match:
        return match.group(4)
    
    # Shortened youtu.be URL
    youtube_short_regex = r'(https?://)?(www\.)?youtu\.be/([^?]+)'
    match = re.match(youtube_short_regex, url)
    if match:
        return match.group(3)
    
    # Parse the URL and extract the video ID from the query string
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    
    return None

# Custom YouTube transcript loader
def load_youtube_transcript(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return None, "Could not extract video ID from URL"
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        if not transcript_list:
            return None, "No transcript available for this video"
        
        # Combine transcript segments into a single text
        transcript_text = " ".join([item['text'] for item in transcript_list])
        
        # Get video metadata
        try:
            response = requests.get(f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}")
            metadata = response.json()
            title = metadata.get('title', 'Unknown Title')
            author = metadata.get('author_name', 'Unknown Author')
        except:
            title = "Unknown Title"
            author = "Unknown Author"
        
        # Create a document with metadata
        from langchain_core.documents import Document
        doc = Document(
            page_content=transcript_text,
            metadata={"source": url, "title": title, "author": author}
        )
        
        return [doc], None
    except Exception as e:
        return None, str(e)

if st.button("Summarize"):
    try:
        # Check all inputs
        if not groq_api_key:
            st.error("Please enter your Groq API Key.")
        elif not URL:
            st.error("Please enter a Youtube video URL or a web page URL.")
        elif not (validators.url(URL) and (URL.startswith("http") or URL.startswith("https"))):
            st.error("Please enter a valid Youtube video URL or a web page URL.")
        else:
            # First identify URL type
            is_youtube = "youtube.com" in URL or "youtu.be" in URL
            
            # Load content based on URL type
            try:
                with st.spinner("Loading Content..."):
                    if is_youtube:
                        documents, error = load_youtube_transcript(URL)
                        if error:
                            st.error(f"Error loading YouTube transcript: {error}")
                            st.stop()
                    else:
                        if not is_url_accessible(URL):
                            st.error(f"Unable to access the URL. Please check that it's correct and publicly accessible.")
                            st.stop()
                            
                        loader = UnstructuredURLLoader(
                            urls=[URL], 
                            ssl_verify=False, 
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                        )
                        documents = loader.load()
                    
                    if not documents or len(documents) == 0:
                        st.error("No content could be extracted from the URL. Please try a different URL.")
                        st.stop()
            except Exception as e:
                st.error(f"Error loading content: {str(e)}")
                st.error(f"Detailed error: {traceback.format_exc()}")
                st.stop()
            
            # Split the document into chunks
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                
                if not docs or len(docs) == 0:
                    st.error("The content could not be properly processed. It might be empty or in an unsupported format.")
                    st.stop()
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
                st.stop()
            
            # Create the Groq model
            try:
                llm = ChatGroq(
                    model=model_name.lower(), 
                    temperature=temperature, 
                    groq_api_key=groq_api_key
                )
            except Exception as e:
                st.error(f"Error initializing Groq API: {str(e)}")
                st.error("Please check your API key and model selection.")
                st.stop()
            
            # Perform summarization based on selected method
            with st.spinner(f"Summarizing Content using {summarization_type}..."):
                try:
                    if summarization_type == "Basic (Stuff)":
                        # Basic approach - combine all text and summarize at once
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=stuff_prompt)
                        summary = chain.run(docs)
                    
                    else:  # Advanced (Map-Reduce)
                        # Advanced approach - summarize each chunk then combine summaries
                        chain = load_summarize_chain(
                            llm,
                            chain_type="map_reduce",
                            map_prompt=map_prompt,
                            combine_prompt=combine_prompt,
                            verbose=True
                        )
                        summary = chain.run(docs)
                except Exception as e:
                    st.error(f"Error during summarization: {str(e)}")
                    # Show more detailed error for debugging
                    st.error(f"Detailed error: {traceback.format_exc()}")
                    st.stop()
            
            # Display the summary
            st.subheader("Summary:")
            st.write(summary)
            
            # Show document stats
            with st.expander("Document Statistics"):
                st.write(f"Number of chunks: {len(docs)}")
                st.write(f"Total content length: {sum(len(doc.page_content) for doc in docs)} characters")
                st.write(f"Summarization method: {summarization_type}")
                
                # If it's a YouTube video, show additional info
                if is_youtube and documents:
                    metadata = documents[0].metadata if hasattr(documents[0], 'metadata') else {}
                    if metadata:
                        if 'title' in metadata:
                            st.write(f"Video Title: {metadata.get('title', 'N/A')}")
                        if 'author' in metadata:
                            st.write(f"Author: {metadata.get('author', 'N/A')}")
            
            st.success("Summary generated successfully!")
            st.balloons()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        st.stop()
