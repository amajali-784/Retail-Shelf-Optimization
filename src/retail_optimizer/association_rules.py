from __future__ import annotations

from itertools import combinations

import pandas as pd

RULE_COLUMNS = ["antecedent", "consequent", "support", "confidence", "lift"]


def mine_association_rules(baskets: pd.DataFrame, min_support: float = 0.015, min_confidence: float = 0.18) -> pd.DataFrame:
    if baskets.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    basket_sets = baskets.groupby("basket_id")["category"].apply(lambda values: sorted(set(values)))
    basket_count = len(basket_sets)
    if basket_count == 0:
        return pd.DataFrame(columns=RULE_COLUMNS)

    item_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}

    for items in basket_sets:
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
        for left, right in combinations(items, 2):
            pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1

    rows = []
    for (left, right), pair_count in pair_counts.items():
        support = pair_count / basket_count
        if support < min_support:
            continue
        for antecedent, consequent in [(left, right), (right, left)]:
            confidence = pair_count / item_counts[antecedent]
            lift = confidence / (item_counts[consequent] / basket_count)
            if confidence >= min_confidence:
                rows.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": round(support, 4),
                        "confidence": round(confidence, 4),
                        "lift": round(lift, 3),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=RULE_COLUMNS)
    return pd.DataFrame(rows, columns=RULE_COLUMNS).sort_values(["lift", "confidence"], ascending=False)
