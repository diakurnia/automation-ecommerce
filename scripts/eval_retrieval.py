"""
Eval-lite for the hybrid search retriever (src/retriever.py).

This was specced in the original design doc's Testing section ("~5-10 query
examples with expected category, to check retrieval quality") but never
actually built during implementation. This closes that gap.

Measures hit-rate@k: for each golden query, does the expected product (or,
for open-ended queries, any product of the expected category) appear in the
top-k results returned by search_products()?

Usage:
    .venv/bin/python scripts/eval_retrieval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import retriever

# Golden set built from the real product IDs currently in the Forme Capsule
# catalog (see products table). Each case targets a specific product by
# descriptive query, except the two open-ended cases which accept any
# product of the expected category (there are 2 rings and 2 bracelets in
# the catalog, so an exact-ID match would be unfairly strict there).
GOLDEN_SET = [
    {
        "query": "chunky gold hoop earrings",
        "expected_id": "gid://shopify/Product/8042395664483",
    },
    {
        "query": "pink sapphire engagement ring rose gold",
        "expected_id": "gid://shopify/Product/8043077173347",
    },
    {
        "query": "diamond tennis bracelet white gold",
        "expected_id": "gid://shopify/Product/8043099947107",
    },
    {
        "query": "pendant necklace white gold moissanite",
        "expected_id": "gid://shopify/Product/8043109056611",
    },
    {
        "query": "gold bangle with champagne crystal",
        "expected_id": "gid://shopify/Product/8045469204579",
    },
    {
        "query": "two-tone halo ring split shank band",
        "expected_id": "gid://shopify/Product/8045474447459",
    },
    {
        "query": "cheapest jewelry piece under 100 dollars",
        "expected_id": "gid://shopify/Product/8042395664483",  # only item under $100
        "max_price": 100,
    },
    {
        "query": "elegant ring for a wedding gift",
        "expected_category": "Rings",
    },
    {
        "query": "everyday bracelet for casual wear",
        "expected_category": "Bracelets",
    },
]

TOP_K = 3


def _hit(case: dict, results: list[dict]) -> bool:
    if "expected_id" in case:
        return any(r["id"] == case["expected_id"] for r in results)
    return any(r["product_type"] == case["expected_category"] for r in results)


def run_eval() -> float:
    rows = []
    for case in GOLDEN_SET:
        results = retriever.search_products(
            case["query"], max_price=case.get("max_price"), top_k=TOP_K)
        hit = _hit(case, results)
        rows.append({
            "query": case["query"],
            "expected": case.get("expected_id") or case.get("expected_category"),
            "hit": hit,
            "top_titles": [r["title"] for r in results[:TOP_K]],
        })

    hit_rate = sum(r["hit"] for r in rows) / len(rows)

    print(f"Retrieval eval-lite: {sum(r['hit'] for r in rows)}/{len(rows)} "
          f"hit@{TOP_K} ({hit_rate:.0%})\n")
    for r in rows:
        mark = "PASS" if r["hit"] else "FAIL"
        print(f"[{mark}] '{r['query']}' -> expected: {r['expected']}")
        if not r["hit"]:
            print(f"       top results: {r['top_titles']}")

    return hit_rate


if __name__ == "__main__":
    run_eval()
