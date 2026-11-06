"""Generate the synthetic BrightMart dataset (200 customers + 4 duplicate rows).

Data-quality problems (missing values, duplicates, invalid values) are inserted on
purpose so the cleaning step in analysis.py can be demonstrated.
The dataset is synthetic: findings illustrate the method, not real BrightMart results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "BrightMart_ML_Dataset.csv"


def generate(seed: int = 42, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Three latent customer groups: Budget (0), Mainstream (1), Premium (2)
    group = rng.choice([0, 1, 2], size=n, p=[0.36, 0.36, 0.28])

    income = np.clip(rng.normal(np.array([290, 540, 890])[group],
                                np.array([70, 90, 90])[group]), 120, 1100).round(0)
    spending = np.clip(rng.normal(np.array([31, 57, 79])[group],
                                  np.array([9, 9, 8])[group]), 5, 99).round(0)
    freq = np.clip(rng.normal(np.array([3.0, 5.3, 8.0])[group], 1.3), 1, 12).round(0)
    age = rng.integers(18, 66, size=n).astype(float)

    disc_p = np.array([[0.2, 0.5, 0.3], [0.3, 0.45, 0.25], [0.5, 0.35, 0.15]])
    discount = np.array(["Low", "Medium", "High"])[
        [rng.choice(3, p=disc_p[g]) for g in group]
    ]

    ad_spend = rng.uniform(5, 50, size=n).round(1)
    sales = (49.6 + 3.5 * ad_spend + rng.normal(0, 20, size=n)).round(1)

    p_resp = np.array([0.03, 0.22, 0.52])[group]
    response = np.where(rng.random(n) < p_resp, "Yes", "No")

    labels = np.array(["Budget", "Standard", "Premium"])
    segment = labels[group]
    flip = rng.random(n) < 0.10  # existing segment labels are only loosely linked
    segment = np.where(flip, rng.choice(labels, size=n), segment)

    df = pd.DataFrame({
        "Customer_ID": [f"C{1000 + i}" for i in range(n)],
        "Age": age,
        "Annual_Income": income,
        "Spending_Score": spending,
        "Purchase_Frequency": freq,
        "Discount_Usage": discount,
        "Customer_Segment": segment,
        "Ad_Spend": ad_spend,
        "Sales": sales,
        "Campaign_Response": response,
    })

    # ---- deliberate data-quality problems ------------------------------------------
    idx = rng.permutation(n)
    df.loc[idx[0:5], "Annual_Income"] = np.nan        # 5 missing income
    df.loc[idx[5:7], "Age"] = np.nan                   # 2 missing age
    df.loc[idx[7], "Age"] = 150                        # 1 impossible age
    df.loc[idx[8:11], "Discount_Usage"] = np.nan       # 3 missing discount usage
    df.loc[idx[11], "Sales"] = -45                     # 1 negative sales
    df.loc[idx[12], "Spending_Score"] = 135            # outside 1-100
    df.loc[idx[13], "Spending_Score"] = 0              # outside 1-100

    dupes = df.sample(4, random_state=seed)            # 4 duplicate rows
    df = pd.concat([df, dupes], ignore_index=True)
    return df


if __name__ == "__main__":
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(data)} raw rows -> {RAW_PATH}")
