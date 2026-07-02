from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src import config
from src.tools import ALL_TOOLS

_cfg = config.load()

SYSTEM = (
    "Kamu asisten belanja yang ramah dan jujur untuk sebuah toko online. "
    "Gunakan tool search_products_tool untuk menemukan produk sesuai kebutuhan user "
    "(termasuk filter harga/kategori/stok bila disebut). Gunakan check_stock_tool / "
    "get_price_tool untuk memastikan stok & harga terkini sebelum merekomendasikan. "
    "Rekomendasikan produk dengan alasan singkat, sebutkan nama & harga. "
    "Ingat preferensi dan budget yang sudah disebut user di percakapan. "
    "Jika tak ada yang cocok, katakan jujur dan sarankan melonggarkan kriteria. "
    "Jawab dalam bahasa yang dipakai user."
)


def build_agent() -> AgentExecutor:
    llm = ChatAnthropic(model=_cfg.anthropic_model, api_key=_cfg.anthropic_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=True,
                         return_intermediate_steps=True)
