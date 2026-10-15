
"""Exploratory Data Analysis for bina.az housing data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_prep import load_raw, clean


FIGURES_DIR = Path("report/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_price_distribution(df):
    """Distribution of housing prices."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["price"], bins=60, edgecolor="white")
    ax.set_xlabel("Price (AZN)")
    ax.set_ylabel("Number of listings")
    ax.set_title("Distribution of Housing Prices")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_distribution.png", dpi=300)
    plt.close(fig)


def plot_log_price_distribution(df):
    """Distribution of log-transformed prices."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(np.log1p(df["price"]), bins=60, edgecolor="white")
    ax.set_xlabel("log(1 + Price)")
    ax.set_ylabel("Number of listings")
    ax.set_title("Log-Transformed Housing Prices")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "log_price_distribution.png", dpi=300)
    plt.close(fig)


def plot_category_prices(df):
    """Median prices by property category."""
    medians = df.groupby("Kateqoriya")["price"].median().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(medians.index, medians.values)

    ax.set_xlabel("Median Price (AZN)")
    ax.set_title("Median Housing Price by Property Category")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_prices.png", dpi=300)
    plt.close(fig)


def plot_area_vs_price(df):
    """Relationship between area and price."""
    sample = df.sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["area_m2"], sample["price"], alpha=0.25, s=12)

    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Price (AZN)")
    ax.set_title("Property Area vs Price")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "area_vs_price.png", dpi=300)
    plt.close(fig)


def plot_location_prices(df):
    """Top 15 locations by listing count, compared by median price."""
    top_locations = df["location"].value_counts().head(15).index

    medians = (
        df[df["location"].isin(top_locations)]
        .groupby("location")["price"]
        .median()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(medians.index, medians.values)

    ax.set_xlabel("Median Price (AZN)")
    ax.set_title("Median Prices in 15 Most Frequent Locations")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "location_prices.png", dpi=300)
    plt.close(fig)

def plot_rooms(df):
    """Listing count and median price by room count."""
    summary = df.groupby("rooms")["price"].agg(["size", "median"])
    labels = [f"{value:g}" for value in summary.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, summary["size"])
    axes[0].set(
        xlabel="Rooms",
        ylabel="Number of listings",
        title="Listings by Room Count",
    )

    axes[1].bar(labels, summary["median"])
    axes[1].set(
        xlabel="Rooms",
        ylabel="Median Price (AZN)",
        title="Median Price by Room Count",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rooms_analysis.png", dpi=200)
    plt.close(fig)


def plot_geographic_prices(df):
    """Geographic price patterns; coordinates are longitude/latitude."""
    sample = df.sample(n=min(10000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))

    points = ax.scatter(
        sample["lng"],
        sample["lat"],
        c=np.log10(sample["price"]),
        cmap="viridis",
        s=9,
        alpha=0.5,
    )
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="Geographic Distribution of Housing Prices",
    )
    fig.colorbar(points, ax=ax, label="log10(Price in AZN)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geographic_prices.png", dpi=200)
    plt.close(fig)


def plot_class_balance(df):
    """Use one TRAIN median threshold for all three splits."""
    from src.data_prep import train_val_test_split, make_tier_label
    import pandas as pd

    prices = df["price"].to_numpy(dtype=float)
    groups = pd.qcut(prices, 4, labels=False)

    _, y_train, _, y_val, _, y_test = train_val_test_split(
        np.arange(len(df)), prices, seed=42, stratify=groups
    )
    _, threshold = make_tier_label(y_train)

    rows = []
    for name, values in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        labels, _ = make_tier_label(values, threshold)
        for label, class_name in [(0, "Standard"), (1, "Premium")]:
            count = int((labels == label).sum())
            rows.append({
                "split": name,
                "class": class_name,
                "count": count,
                "percent": 100 * count / len(labels),
                "threshold_AZN": threshold,
            })

    table = pd.DataFrame(rows)
    tables_dir = Path("report/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "eda_class_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(3)
    for offset, class_name in [(-0.2, "Standard"), (0.2, "Premium")]:
        values = table.loc[table["class"] == class_name, "percent"]
        bars = ax.bar(
            positions + offset, values, width=0.4, label=class_name
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3)

    ax.set_xticks(positions, ["Train", "Validation", "Test"])
    ax.set(
        ylabel="Share within split (%)",
        title=f"Class Balance — Train Threshold: {threshold:,.0f} AZN",
        ylim=(0, 100),
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=200)
    plt.close(fig)
    print(table.round(2).to_string(index=False))
def main():
    df = clean(load_raw())

    print("Cleaned dataset:", df.shape)
    print("Median price:", df["price"].median())
    print("Mean price:", df["price"].mean())

    plot_price_distribution(df)
    plot_log_price_distribution(df)
    plot_category_prices(df)
    plot_area_vs_price(df)
    plot_location_prices(df)
    plot_rooms(df)
    plot_geographic_prices(df)
    plot_class_balance(df)

    print("All EDA figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
