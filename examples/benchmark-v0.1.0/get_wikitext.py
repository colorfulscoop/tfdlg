from pathlib import Path
from urllib.request import urlretrieve
import zipfile


def get_wikitext(level):
    """
    Download Wikitext-2 https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    """
    assert level in ["2_raw", "2_word", "103_raw", "103_word"]

    level_url = {
        "2_raw": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
        "2_word": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
        "103_raw": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
        "103_word": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
    }
    level_filename = {
        "2_raw": "  wikitext-2-raw.zip",
        "2_word":   "wikitext-2.zip",
        "103_raw":  "wikitext-103-raw.zip",
        "103_word": "wikitext-103.zip",
    }
    url = level_url[level]
    filename = level_filename[level]
    filepath, _ = urlretrieve(url, filename)
    print("Download url:", url)
    print("Downloaded file path:", filepath)

    path_to_unzip = Path(filepath).parent
    print("Unarchived and placed under", path_to_unzip)
    with zipfile.ZipFile(filepath) as zip_:
        zip_.extractall(path_to_unzip)


if __name__ == "__main__":
    import sys

    level = sys.argv[1] if len(sys.argv) >= 2 else "raw"
    get_wikitext(level)
