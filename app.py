import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import build_agent

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️")
st.title("🛍️ AI Shopping Assistant")

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (role, content)


def _extract_text(output) -> str:
    """Claude's tool-calling responses can return a list of content blocks
    instead of a plain string; normalize to text either way."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(
            block.get("text", "") for block in output
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(output)


def _render_product_cards(result):
    for action, observation in result.get("intermediate_steps", []):
        if action.tool != "search_products_tool":
            continue
        for line in str(observation).splitlines():
            if not line.startswith("["):
                continue
            # format: [id] title | CUR price | type | stok: x | url
            try:
                _, rest = line.split("] ", 1)
                title, cur_price, _type, _stok, url = [s.strip() for s in rest.split("|")]
            except ValueError:
                continue
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(cur_price)
                if url:
                    st.markdown(f"[Lihat produk]({url})")


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Tanya apa saja soal produk..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    history = []
    for role, content in st.session_state.messages[:-1]:
        history.append(HumanMessage(content=content) if role == "user"
                       else AIMessage(content=content))

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            result = st.session_state.agent.invoke({
                "input": prompt,
                "chat_history": history,
            })
        answer_text = _extract_text(result["output"])
        st.markdown(answer_text)
        _render_product_cards(result)
    st.session_state.messages.append(("assistant", answer_text))
