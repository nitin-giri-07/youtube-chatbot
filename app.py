
import re
import hashlib
import uuid

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube AI Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 YouTube AI Analyzer")
st.write(
    "Summarize YouTube videos and ask questions "
    "about their transcripts using AI."
)


# --------------------------------------------------
# 2. LET EACH USER ENTER THEIR OWN GROQ API KEY
# --------------------------------------------------

st.sidebar.title("🔑 API Settings")

st.sidebar.write(
    "Enter your own Groq API key to use the AI features."
)

st.sidebar.markdown(
    "[Get a Groq API key](https://console.groq.com/keys)"
)

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="Paste your API key here"
)

st.sidebar.caption(
    "Use your own key. Never share it with other people."
)

if not api_key.strip():
    st.info(
        "Enter your Groq API key in the sidebar "
        "to start using the analyzer."
    )
    st.stop()


# Create the language model using the current user's key.
# The key is not written to a file or database by this code.

try:
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key.strip(),
        temperature=0
    )
except Exception as error:
    st.error("Could not initialize the language model.")
    st.caption(str(error))
    st.stop()


# --------------------------------------------------
# 3. LOAD THE LOCAL EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 4. EXTRACT THE VIDEO ID FROM A URL OR VIDEO ID
# --------------------------------------------------

def extract_video_id(value):
    value = value.strip()

    # The user may enter the 11-character video ID directly.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    # Supports standard YouTube URLs, youtu.be links,
    # Shorts URLs, and embed URLs.
    match = re.search(
        r"(?:youtube\.com/(?:watch\?.*?v=|embed/|shorts/)|youtu\.be/)"
        r"([A-Za-z0-9_-]{11})",
        value
    )

    if match:
        return match.group(1)

    return None


# --------------------------------------------------
# 5. FETCH THE VIDEO TRANSCRIPT
# --------------------------------------------------

st.subheader("1. Load a YouTube video")

video_input = st.text_input(
    "YouTube video URL or video ID",
    placeholder="https://www.youtube.com/watch?v=..."
)

language = st.selectbox(
    "Transcript language",
    options=["en", "hi"],
    format_func=lambda code: (
        "English" if code == "en" else "Hindi"
    )
)

if st.button("Load Video", type="primary"):
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error("Please enter a valid YouTube URL or video ID.")

    else:
        try:
            with st.spinner("Fetching video transcript..."):

                transcript_result = YouTubeTranscriptApi().fetch(
                    video_id,
                    languages=[language]
                )

                snippets = transcript_result.snippets

                transcript_text = "\n".join(
                    snippet.text for snippet in snippets
                )

                if not transcript_text.strip():
                    st.error("The transcript is empty.")

                else:
                    # Save the results for later Streamlit reruns.
                    st.session_state["video_id"] = video_id
                    st.session_state["transcript"] = transcript_text

                    st.session_state["snippets"] = [
                        {
                            "text": snippet.text,
                            "start": snippet.start,
                            "duration": snippet.duration
                        }
                        for snippet in snippets
                    ]

                    # Reset results from any previously loaded video.
                    st.session_state["vector_store"] = None
                    st.session_state["summary"] = None
                    st.session_state["summary_truncated"] = False

                    st.success("Transcript loaded successfully!")

        except Exception as error:
            st.error(
                "Could not fetch the transcript. "
                "The video may not have captions in the selected language."
            )
            st.caption(f"Technical details: {error}")


# --------------------------------------------------
# 6. DISPLAY THE LOADED VIDEO AND TRANSCRIPT
# --------------------------------------------------

if "transcript" in st.session_state:

    transcript = st.session_state["transcript"]
    video_id = st.session_state["video_id"]

    st.divider()
    st.subheader("2. Video details")

    st.video(
        f"https://www.youtube.com/watch?v={video_id}"
    )

    st.metric(
        "Transcript characters",
        f"{len(transcript):,}"
    )

    with st.expander("View transcript"):
        st.text(transcript[:30000])

        if len(transcript) > 30000:
            st.caption(
                "Only the first 30,000 characters are displayed here."
            )


    # --------------------------------------------------
    # 7. CREATE DOCUMENTS FOR RETRIEVAL
    # --------------------------------------------------

    if st.session_state.get("vector_store") is None:

        try:
            with st.spinner(
                "Preparing transcript chunks and search index..."
            ):

                snippets = st.session_state["snippets"]

                # Group nearby captions into approximately
                # 60-second transcript sections.
                documents = []
                current_text = []
                section_start = None

                for snippet in snippets:

                    if section_start is None:
                        section_start = snippet["start"]

                    # Start a new section when the current section
                    # has reached approximately 60 seconds.
                    if (
                        current_text
                        and snippet["start"] - section_start >= 60
                    ):
                        documents.append(
                            Document(
                                page_content=" ".join(current_text),
                                metadata={
                                    "video_id": video_id,
                                    "start": section_start
                                }
                            )
                        )

                        current_text = []
                        section_start = snippet["start"]

                    current_text.append(snippet["text"])

                # Save the final section.
                if current_text:
                    documents.append(
                        Document(
                            page_content=" ".join(current_text),
                            metadata={
                                "video_id": video_id,
                                "start": section_start
                            }
                        )
                    )

                # Split long sections into smaller chunks.
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = splitter.split_documents(documents)

                if not chunks:
                    st.error("No transcript chunks were created.")
                    st.stop()

                # Make a unique in-memory collection for this load.
                video_hash = hashlib.sha1(
                    video_id.encode("utf-8")
                ).hexdigest()[:10]

                collection_name = (
                    f"yt_{video_hash}_{uuid.uuid4().hex[:8]}"
                )

                # Convert chunks into embeddings and store them
                # in ChromaDB.
                st.session_state["vector_store"] = (
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=load_embeddings(),
                        collection_name=collection_name
                    )
                )

                st.session_state["chunk_count"] = len(chunks)

        except Exception as error:
            st.error("Could not prepare the transcript search index.")
            st.caption(f"Technical details: {error}")
            st.stop()

    st.success(
        f"Search index ready: "
        f"{st.session_state['chunk_count']} transcript chunks"
    )


    # --------------------------------------------------
    # 8. GENERATE A VIDEO SUMMARY
    # --------------------------------------------------

    st.divider()
    st.subheader("3. Generate a summary")

    if st.button("Generate Summary"):

        summary_input = transcript[:24000]

        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a helpful YouTube video analyst.

                Summarize the supplied transcript in clear language.

                Include:
                1. The main topic
                2. Important ideas and explanations
                3. Key takeaways
                4. Important names, facts, or examples when present

                Do not invent facts that are not in the transcript.
                If the transcript is incomplete, mention that limitation.
                """
            ),
            (
                "human",
                "Analyze this YouTube transcript:\n\n{transcript}"
            )
        ])

        try:
            with st.spinner("Generating summary..."):

                response = (
                    summary_prompt | llm
                ).invoke({
                    "transcript": summary_input
                })

                st.session_state["summary"] = response.content

                st.session_state["summary_truncated"] = (
                    len(transcript) > len(summary_input)
                )

        except Exception as error:
            st.error(
                "Summary generation failed. Check your API key, "
                "model availability, quota, and internet connection."
            )
            st.caption(f"Technical details: {error}")

    if st.session_state.get("summary"):

        if st.session_state.get("summary_truncated"):
            st.warning(
                "The transcript was long, so the summary used "
                "only its first 24,000 characters."
            )

        st.markdown("### Video Summary")
        st.markdown(st.session_state["summary"])


    # --------------------------------------------------
    # 9. ASK QUESTIONS USING RAG
    # --------------------------------------------------

    st.divider()
    st.subheader("4. Ask questions about the video")

    question = st.text_input(
        "Your question",
        key="video_question",
        placeholder="What are the main points discussed in this video?"
    )

    if st.button("Ask Question"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Searching the transcript and generating an answer..."
                ):

                    # Find the transcript chunks most relevant
                    # to the user's question.
                    retriever = (
                        st.session_state["vector_store"]
                        .as_retriever(
                            search_kwargs={"k": 5}
                        )
                    )

                    retrieved_docs = retriever.invoke(question)

                    context = "\n\n".join(
                        (
                            f"Approximate section start: "
                            f"{doc.metadata.get('start', 0):.1f} seconds\n"
                            f"Transcript: {doc.page_content}"
                        )
                        for doc in retrieved_docs
                    )

                    qa_prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """
                            You answer questions about a YouTube video.

                            Use only the transcript excerpts supplied below.
                            If the excerpts do not contain enough information,
                            say that you cannot determine the answer from
                            the retrieved transcript.

                            Do not invent details or claim that an
                            approximate section timestamp is exact.
                            """
                        ),
                        (
                            "human",
                            """
                            Question:
                            {question}

                            Relevant transcript excerpts:
                            {context}
                            """
                        )
                    ])

                    answer = (
                        qa_prompt | llm
                    ).invoke({
                        "question": question,
                        "context": context
                    })

                    st.markdown("### Answer")
                    st.write(answer.content)

                    with st.expander(
                        "View retrieved transcript excerpts"
                    ):

                        for doc in retrieved_docs:

                            start = doc.metadata.get("start", 0)

                            st.markdown(
                                f"**Approximate section start: "
                                f"{start:.1f} seconds**"
                            )

                            st.write(doc.page_content)

            except Exception as error:
                st.error(
                    "Question answering failed. Check your API key, "
                    "model availability, and connection."
                )
                st.caption(f"Technical details: {error}")


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "YouTube AI Analyzer | Transcript availability depends on YouTube. "
    "AI answers may be inaccurate."
)
