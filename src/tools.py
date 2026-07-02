from langchain_core.tools import tool
from src import retriever, shopify_client


@tool
def search_products_tool(query: str, min_price: float | None = None,
                         max_price: float | None = None,
                         category: str | None = None,
                         only_in_stock: bool = False) -> str:
    """Cari produk toko berdasarkan makna query + filter harga/kategori/stok.
    Gunakan saat user ingin menemukan atau membandingkan produk."""
    hits = retriever.search_products(
        query, min_price=min_price, max_price=max_price,
        category=category, only_in_stock=only_in_stock)
    if not hits:
        return "Tidak ada produk yang cocok."
    lines = []
    for h in hits:
        stok = "tersedia" if h["in_stock"] else "habis"
        lines.append(
            f"[{h['id']}] {h['title']} | {h['currency']} {h['price']:.0f} | "
            f"{h['product_type']} | stok: {stok} | {h['url']}")
    return "\n".join(lines)


@tool
def check_stock_tool(product_id: str) -> str:
    """Cek jumlah stok terkini sebuah produk berdasarkan product_id (gid)."""
    p = shopify_client.get_product(product_id)
    return f"{p['title']}: stok {p['inventory_qty']} unit."


@tool
def get_price_tool(product_id: str) -> str:
    """Cek harga terkini sebuah produk berdasarkan product_id (gid)."""
    p = shopify_client.get_product(product_id)
    return f"{p['title']}: harga {p['price']:.0f}."


ALL_TOOLS = [search_products_tool, check_stock_tool, get_price_tool]
