import requests
from src import config

_cfg = config.load()

_SHOP_QUERY = """
{ shop { currencyCode } }
"""

_PRODUCTS_QUERY = """
query($cursor: String) {
  products(first: 50, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id title description productType tags onlineStoreUrl
        featuredImage { url }
        variants(first: 1) {
          edges { node { price inventoryQuantity } }
        }
      }
    }
  }
}
"""

_PRODUCT_QUERY = """
query($id: ID!) {
  product(id: $id) {
    id title
    variants(first: 1) { edges { node { price inventoryQuantity } } }
  }
}
"""


def _post_graphql(query: str, variables: dict) -> dict:
    url = f"https://{_cfg.shopify_domain}/admin/api/{_cfg.shopify_api_version}/graphql.json"
    resp = requests.post(
        url,
        json={"query": query, "variables": variables},
        headers={"X-Shopify-Access-Token": _cfg.shopify_token,
                 "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_shop_currency() -> str:
    data = _post_graphql(_SHOP_QUERY, {})
    return data["data"]["shop"]["currencyCode"]


def fetch_products() -> list[dict]:
    """Ambil semua produk (paginasi) sebagai list dict mentah."""
    out, cursor = [], None
    while True:
        data = _post_graphql(_PRODUCTS_QUERY, {"cursor": cursor})
        conn = data["data"]["products"]
        for edge in conn["edges"]:
            node = edge["node"]
            node["variants"] = [v["node"] for v in node["variants"]["edges"]]
            out.append(node)
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    return out


def get_product(product_id: str) -> dict:
    data = _post_graphql(_PRODUCT_QUERY, {"id": product_id})
    node = data["data"]["product"]
    v = node["variants"]["edges"][0]["node"] if node["variants"]["edges"] else {}
    return {
        "id": node["id"],
        "title": node["title"],
        "price": float(v.get("price") or 0),
        "inventory_qty": int(v.get("inventoryQuantity") or 0),
    }
