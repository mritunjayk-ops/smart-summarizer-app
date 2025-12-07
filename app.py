import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
st.subheader("Enter a YouTube video URL or a web page URL to get started.")

with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    st.subheader("Summary Settings")
    summarization_type = st.radio(
        "Summary mode",
        ["Basic (fast)", "Advanced (better for long content)"],
        index=0,
    )
    model_name = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "openai/gpt-oss-120b",
        ],
        index=0,
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.0, 0.1)

URL = st.text_input("Paste YouTube or web page URL", label_visibility="collapsed")

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


def is_url_accessible(url: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.head(url, headers=headers, timeout=5)
        return 200 <= response.status_code < 400
    except Exception:
        return False


def extract_video_id(url: str):
    youtube_regex = r"(https?://)?(www\.)?youtube\.(com|nl)/watch\?v=([^&]+)"
    match = re.match(youtube_regex, url)
    if match:
        return match.group(4)

    youtube_short_regex = r"(https?://)?(www\.)?youtu\.be/([^?]+)"
    match = re.match(youtube_short_regex, url)
    if match:
        return match.group(3)

    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    return None


@st.cache_data(show_spinner=False)
def cached_youtube_transcript(url: str):
    return _load_youtube_transcript(url)


def _load_youtube_transcript(url: str):
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


@st.cache_data(show_spinner=False)
def cached_web_documents(url: str):
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    return loader.load()


def advanced_summarize(llm, docs, map_prompt: PromptTemplate, combine_prompt: PromptTemplate, max_chunks: int = 12) -> str:
    selected_docs = docs[:max_chunks]
    summaries = []

    for doc in selected_docs:
        text = doc.page_content
        prompt_text = map_prompt.format(text=text)
        res = llm.invoke(prompt_text)
        summaries.append(res.content)

    combined_text = "\n\n".join(summaries)
    final_prompt = combine_prompt.format(text=combined_text)
    final_res = llm.invoke(final_prompt)
    return final_res.content


if st.button("Summarize"):
    try:
        if not groq_api_key:
            st.error("Please enter your Groq API Key.")
        elif not URL:
            st.error("Please enter a YouTube video URL or a web page URL.")
        elif not (validators.url(URL) and (URL.startswith("http://") or URL.startswith("https://"))):
            st.error("Please enter a valid URL starting with http or https.")
        else:
            is_youtube = "youtube.com" in URL or "youtu.be" in URL

            try:
                with st.spinner("Loading content..."):
                    if is_youtube:
                        documents, error = cached_youtube_transcript(URL)
                        if error:
                            st.error(f"Error loading YouTube transcript: {error}")
                            st.stop()
                    else:
                        if not is_url_accessible(URL):
                            st.error("Unable to access the URL. Please check that it is correct and publicly accessible.")
                            st.stop()
                        documents = cached_web_documents(URL)

                    if not documents:
                        st.error("No content could be extracted.")
                        st.stop()
            except Exception as e:
                st.error(f"Error loading content: {str(e)}")
                st.error(traceback.format_exc())
                st.stop()

            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                )
                docs = text_splitter.split_documents(documents)
                if not docs:
                    st.error("Failed to process content.")
                    st.stop()
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
                st.stop()

            total_len = sum(len(doc.page_content) for doc in docs)
            use_advanced = (
                summarization_type == "Advanced (better for long content)"
                and total_len > 8000
            )

            try:
                llm = ChatGroq(
                    model=model_name,
                    temperature=temperature,
                    groq_api_key=groq_api_key,
                )
            except Exception as e:
                st.error(f"Error initializing Groq API: {str(e)}")
                st.stop()

            with st.spinner("Summarizing..."):
                try:
                    if not use_advanced:
                        chain = load_summarize_chain(
                            llm,
                            chain_type="stuff",
                            prompt=stuff_prompt,
                        )
                        summary = chain.run(docs)
                    else:
                        summary = advanced_summarize(
                            llm,
                            docs,
                            map_prompt=map_prompt,
                            combine_prompt=combine_prompt,
                        )
                except Exception as e:
                    st.error(f"Error during summarization: {str(e)}")
                    st.error(traceback.format_exc())
                    st.stop()

            st.subheader("Summary")
            st.write(summary)

            with st.expander("Details"):
                st.write(f"Number of chunks: {len(docs)}")
                st.write(f"Total content length: {total_len} characters")
                st.write(
                    "Engine used: "
                    + ("Advanced (custom map-reduce)" if use_advanced else "Basic (stuff)")
                )

                if is_youtube and documents:
                    metadata = getattr(documents[0], "metadata", {}) or {}
                    title = metadata.get("title")
                    author = metadata.get("author")
                    if title:
                        st.write(f"Video Title: {title}")
                    if author:
                        st.write(f"Author: {author}")

            st.success("Summary generated!")
            st.balloons()

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()
