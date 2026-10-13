
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

    print("All EDA figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
