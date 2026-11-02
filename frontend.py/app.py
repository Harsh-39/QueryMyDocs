import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    API_URL = st.secrets["API_URL"]
except (KeyError, StreamlitSecretNotFoundError):
    API_URL = "http://localhost:10000"

st.set_page_config(page_title="QueryMyDocs", layout="wide")
st.title("📄 QueryMyDocs (Chat RAG)")

if "docs" not in st.session_state:
    st.session_state.docs = {}

if "messages" not in st.session_state:
    st.session_state.messages = {}

if "uploading_file" not in st.session_state:
    st.session_state.uploading_file = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    filename = uploaded_file.name
    if filename not in st.session_state.docs:
        with st.spinner(f"Uploading and indexing **{filename}**..."):
            files = {
                "file": (
                    filename,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            try:
                res = requests.post(f"{API_URL}/upload", files=files)
            except requests.RequestException as e:
                st.error(f"Could not upload the PDF: {e}")
            else:
                if res.status_code == 200:
                    data = res.json()
                    doc_id = data["doc_id"]
                    st.session_state.docs[filename] = doc_id
                    st.session_state.messages[filename] = []
                    st.success(
                        f"✅ Indexed **{filename}** ({data['chunks']} chunks)"
                    )
                else:
                    st.error(res.text)
    else:
        st.info(f"**{filename}** is already indexed. Select it below.")

if st.session_state.docs:
    selected_doc = st.selectbox(
        "🗂️ Select Document to Chat With",
        list(st.session_state.docs.keys()),
    )
    doc_id = st.session_state.docs[selected_doc]

    if selected_doc not in st.session_state.messages:
        st.session_state.messages[selected_doc] = []

    st.divider()

    for msg in st.session_state.messages[selected_doc]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input(f"Ask something about {selected_doc}...")
    if query:
        st.session_state.messages[selected_doc].append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{API_URL}/ask",
                        params={"query": query, "doc_id": doc_id},
                    )
                except requests.RequestException as e:
                    st.error(f"Could not reach the API: {e}")
                else:
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No answer")
                        docs = data.get("docs", [])
                        st.markdown(answer)

                        if docs:
                            with st.expander("📚 Sources"):
                                for i, d in enumerate(docs):
                                    st.markdown(f"**[Source {i + 1}]**")
                                    st.caption(d[:500])
                                    st.divider()

                        st.session_state.messages[selected_doc].append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        st.error(res.text)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Clear Chat History"):
            selected = selected_doc
            st.session_state.messages[selected] = []
            st.rerun()

    with col2:
        if st.button("🗑️ Remove All Documents"):
            st.session_state.docs = {}
            st.session_state.messages = {}
            st.rerun()
