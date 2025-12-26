import streamlit as st
from rag import RAG

st.set_page_config(page_title="RAG Cars", layout="centered")
st.title("Chatbot RAG de Autos")

@st.cache_resource
def load_rag():
    return RAG()

rag = load_rag()

with st.sidebar:
    st.subheader("Parámetros del Modelo")
    topk = st.slider("Top-k chunks", 1, 10, 5, 1)
    max_tokens = st.slider("Max tokens salida", 64, 1024, 400, 32)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    show_sources = st.checkbox("Mostrar fuentes", True)

    if st.button("Limpiar chat"):
        st.session_state.messages = []
        st.session_state.usage = []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "usage" not in st.session_state:
    st.session_state.usage = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Pregunta sobre fichas técnicas...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            ans, cites, usage = rag.ask(
                q, topk=topk, max_tokens=max_tokens, temperature=temperature
            )
            st.markdown(ans)

            if usage:
                st.caption(
                    f"Tokens — prompt: {usage.prompt_tokens} | output: {usage.completion_tokens} | total: {usage.total_tokens}"
                )
                st.session_state.usage.append(usage.total_tokens)

            if show_sources and cites:
                st.markdown("**Fuentes:**")
                for c in cites:
                    st.write(f"- {c}")

    st.session_state.messages.append({"role": "assistant", "content": ans})

if st.session_state.usage:
    st.divider()
    st.subheader("Consumo acumulado (sesión)")
    st.write(f"Total tokens: {sum(st.session_state.usage)}")