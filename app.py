import validators
import streamlit as st
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

st.set_page_config(
    page_title="LangChain Groq",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦙LangChain: Summarize Youtube Videos and Web Pages")
st.subheader("Please enter a Youtube video URL or a web page URL to get started.")

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
        [
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "qwen/qwen3-32b",
            "openai/gpt-oss-120b",
        ],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

URL = st.text_input("Enter a Youtube video URL or a web page URL", label_visibility="collapsed")

map_prompt_template = """Write a concise summary of the following content:
{text}
CONCISE SUMMARY:"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """Combine these summaries into a single comprehensive summary of about 300 words:
{text}
COMPREHENSIVE SUMMARY:"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

stuff_prompt_template = """Summarize the following content in about 300 words:
Content: {text}
SUMMARY:"""
stuff_prompt = PromptTemplate(template=stuff_prompt_template, input_variables=["text"])


def is_url_accessible(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.head(url, headers=headers, timeout=5)
        return 200 <= response.status_code < 400
    except Exception:
        return False


def extract_video_id(url):
    youtube_regex = r'(https?://)?(www\.)?youtube\.(com|nl)/watch\?v=([^&]+)'
    match = re.match(youtube_regex, url)
    if match:
        return match.group(4)

    youtube_short_regex = r'(https?://)?(www\.)?youtu\.be/([^?]+)'
    match = re.match(youtube_short_regex, url)
    if match:
        return match.group(3)

    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    return None


def load_youtube_transcript(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return None, "Could not extract video ID from URL"

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])
        segments = transcript.to_raw_data()
        if not segments:
            return None, "No transcript available for this video"

        transcript_text = " ".join(item["text"] for item in segments)

        try:
            response = requests.get(
                f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
            )
            metadata = response.json()
            title = metadata.get("title", "Unknown Title")
            author = metadata.get("author_name", "Unknown Author")
        except Exception:
            title = "Unknown Title"
            author = "Unknown Author"

        from langchain_core.documents import Document

        doc = Document(
            page_content=transcript_text,
            metadata={"source": url, "title": title, "author": author},
        )

        return [doc], None
    except Exception as e:
        msg = str(e)
        if "IP has been blocked" in msg or "RequestBlocked" in msg or "IpBlocked" in msg:
            return None, "YouTube is blocking transcript requests from this server. Try running the app on your own machine instead."
        if "Could not retrieve a transcript" in msg:
            return None, "No transcript could be retrieved for this video. It may have transcripts disabled or be blocked."
        return None, msg


if st.button("Summarize"):
    try:
        if not groq_api_key:
            st.error("Please enter your Groq API Key.")
        elif not URL:
            st.error("Please enter a Youtube video URL or a web page URL.")
        elif not (validators.url(URL) and (URL.startswith("http") or URL.startswith("https"))):
            st.error("Please enter a valid URL.")
        else:
            is_youtube = "youtube.com" in URL or "youtu.be" in URL

            try:
                with st.spinner("Loading Content..."):
                    if is_youtube:
                        documents, error = load_youtube_transcript(URL)
                        if error:
                            st.error(f"Error loading YouTube transcript: {error}")
                            st.stop()
                    else:
                        if not is_url_accessible(URL):
                            st.error("Unable to access the URL.")
                            st.stop()

                        loader = UnstructuredURLLoader(
                            urls=[URL],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        documents = loader.load()

                    if not documents:
                        st.error("No content could be extracted.")
                        st.stop()
            except Exception as e:
                st.error(f"Error loading content: {str(e)}")
                st.error(f"{traceback.format_exc()}")
                st.stop()

            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                docs = text_splitter.split_documents(documents)
                if not docs:
                    st.error("Failed to process content.")
                    st.stop()
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
                st.stop()

            try:
                llm = ChatGroq(
                    model=model_name,
                    temperature=temperature,
                    groq_api_key=groq_api_key,
                )
            except Exception as e:
                st.error(f"Error initializing Groq API: {str(e)}")
                st.stop()

            with st.spinner(f"Summarizing Content using {summarization_type}..."):
                try:
                    if summarization_type == "Basic (Stuff)":
                        chain = load_summarize_chain(
                            llm,
                            chain_type="stuff",
                            prompt=stuff_prompt,
                        )
                        summary = chain.run(docs)
                    else:
                        chain = load_summarize_chain(
                            llm,
                            chain_type="map_reduce",
                            map_prompt=map_prompt,
                            combine_prompt=combine_prompt,
                            verbose=True,
                        )
                        summary = chain.run(docs)
                except Exception as e:
                    st.error(f"Error during summarization: {str(e)}")
                    st.error(f"{traceback.format_exc()}")
                    st.stop()

            st.subheader("Summary:")
            st.write(summary)

            with st.expander("Document Statistics"):
                st.write(f"Number of chunks: {len(docs)}")
                total_len = sum(len(doc.page_content) for doc in docs)
                st.write(f"Total content length: {total_len}")
                st.write(f"Summarization method: {summarization_type}")

                if is_youtube and documents:
                    metadata = getattr(documents[0], "metadata", {}) or {}
                    title = metadata.get("title")
                    author = metadata.get("author")
                    if title:
                        st.write(f"Video Title: {title}")
                    if author:
                        st.write(f"Author: {author}")

            st.success("Summary generated successfully!")
            st.balloons()

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.error(f"{traceback.format_exc()}")
        st.stop()
