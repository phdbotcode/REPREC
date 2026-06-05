"""Download Amazon review data from Stanford SNAP."""

import os
import gzip
import json
import urllib.request
from typing import Dict, List

from tqdm import tqdm

SNAP_URLS = {
    "Beauty": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Beauty_5.json.gz"
    ),
    "Sports_and_Outdoors": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Sports_and_Outdoors_5.json.gz"
    ),
    "Toys_and_Games": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Toys_and_Games_5.json.gz"
    ),
    "Movies_and_TV": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Movies_and_TV_5.json.gz"
    ),
    "Tools_and_Home_Improvement": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Tools_and_Home_Improvement_5.json.gz"
    ),
    "Automotive": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Automotive_5.json.gz"
    ),
    "Electronics": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Electronics_5.json.gz"
    ),
    "Home_and_Kitchen": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Home_and_Kitchen_5.json.gz"
    ),
    "Pet_Supplies": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Pet_Supplies_5.json.gz"
    ),
    "Office_Products": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Office_Products_5.json.gz"
    ),
    "Musical_Instruments": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Musical_Instruments_5.json.gz"
    ),
    "Patio_Lawn_and_Garden": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Patio_Lawn_and_Garden_5.json.gz"
    ),
    "Industrial_and_Scientific": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_Industrial_and_Scientific_5.json.gz"
    ),
    "CDs_and_Vinyl": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/reviews_CDs_and_Vinyl_5.json.gz"
    ),
}

META_URLS = {
    "Beauty": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Beauty.json.gz"
    ),
    "Sports_and_Outdoors": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Sports_and_Outdoors.json.gz"
    ),
    "Toys_and_Games": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Toys_and_Games.json.gz"
    ),
    "Movies_and_TV": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Movies_and_TV.json.gz"
    ),
    "Tools_and_Home_Improvement": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Tools_and_Home_Improvement.json.gz"
    ),
    "Automotive": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Automotive.json.gz"
    ),
    "Electronics": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Electronics.json.gz"
    ),
    "Home_and_Kitchen": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Home_and_Kitchen.json.gz"
    ),
    "Pet_Supplies": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Pet_Supplies.json.gz"
    ),
    "Office_Products": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Office_Products.json.gz"
    ),
    "Musical_Instruments": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Musical_Instruments.json.gz"
    ),
    "Patio_Lawn_and_Garden": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Patio_Lawn_and_Garden.json.gz"
    ),
    "Industrial_and_Scientific": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_Industrial_and_Scientific.json.gz"
    ),
    "CDs_and_Vinyl": (
        "http://snap.stanford.edu/data/amazon/productGraph/"
        "categoryFiles/meta_CDs_and_Vinyl.json.gz"
    ),
}


def download_file(url: str, save_path: str) -> bool:
    """Download a file with a tqdm progress bar."""
    try:
        with tqdm(unit="B", unit_scale=True, desc=os.path.basename(save_path)) as pbar:

            def reporthook(block_num, block_size, total_size):
                if pbar.total is None and total_size > 0:
                    pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(url, save_path, reporthook)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def decompress_gz(gz_path: str, output_path: str) -> None:
    """Decompress a .gz file and remove the archive."""
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(gz_path)


def parse_json_line(line: str) -> dict:
    """Parse a single JSON line, falling back to eval for malformed JSON."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        try:
            return eval(line.strip())  # noqa: S307 – SNAP files need this
        except Exception:
            return None


def load_reviews(file_path: str) -> List[dict]:
    """Load reviews from a line-delimited JSON file."""
    reviews = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            review = parse_json_line(line)
            if review:
                reviews.append(
                    {
                        "user_id": review.get("reviewerID"),
                        "item_id": review.get("asin"),
                        "rating": review.get("overall", 0),
                        "timestamp": review.get("unixReviewTime", 0),
                        "text": review.get("reviewText", ""),
                        "summary": review.get("summary", ""),
                    }
                )
    return reviews


def load_metadata(file_path: str) -> Dict[str, dict]:
    """Load item metadata keyed by ASIN."""
    metadata = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            item = parse_json_line(line)
            if item:
                asin = item.get("asin")
                if asin:
                    metadata[asin] = {
                        "title": item.get("title", ""),
                        "category": (
                            item.get("categories", [[""]])[0][0]
                            if item.get("categories")
                            else ""
                        ),
                        "price": item.get("price", ""),
                        "brand": item.get("brand", ""),
                    }
    return metadata


def download_amazon_data(
    data_dir: str = "./data/raw",
    categories: List[str] = None,
    download_meta: bool = True,
) -> Dict[str, dict]:
    """Download and parse Amazon review + metadata for given categories.

    Returns
    -------
    dict with keys ``reviews`` and ``meta``, each mapping category → data.
    """
    if categories is None:
        categories = list(SNAP_URLS.keys())

    os.makedirs(data_dir, exist_ok=True)
    data: Dict[str, dict] = {"reviews": {}, "meta": {}}

    for category in categories:
        if category not in SNAP_URLS:
            print(f"Category {category} not available, skipping...")
            continue

        # ── Reviews ──────────────────────────────────────────────────────
        review_gz = os.path.join(data_dir, f"{category}_reviews.json.gz")
        review_path = os.path.join(data_dir, f"{category}_reviews.json")

        if not os.path.exists(review_path):
            print(f"Downloading {category} reviews...")
            if download_file(SNAP_URLS[category], review_gz):
                print(f"Decompressing {category} reviews...")
                decompress_gz(review_gz, review_path)

        if os.path.exists(review_path):
            print(f"Loading {category} reviews...")
            data["reviews"][category] = load_reviews(review_path)
            print(f"  Loaded {len(data['reviews'][category])} reviews")

        # ── Metadata ─────────────────────────────────────────────────────
        if download_meta and category in META_URLS:
            meta_gz = os.path.join(data_dir, f"{category}_meta.json.gz")
            meta_path = os.path.join(data_dir, f"{category}_meta.json")

            if not os.path.exists(meta_path):
                print(f"Downloading {category} metadata...")
                if download_file(META_URLS[category], meta_gz):
                    print(f"Decompressing {category} metadata...")
                    decompress_gz(meta_gz, meta_path)

            if os.path.exists(meta_path):
                print(f"Loading {category} metadata...")
                data["meta"][category] = load_metadata(meta_path)
                print(f"  Loaded {len(data['meta'][category])} items")

    return data
