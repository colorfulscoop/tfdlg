from pathlib import Path
from urllib.request import urlretrieve
import zipfile


def get_wikitext2(level):
    """
    Download Wikitext-2 https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    """
    assert level in ["raw", "word"]

    level_url = {
        "raw": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
        "word": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    }
    level_filename = {
        "raw": "wikitext-2-raw.zip",
        "word": "wikitext-2.zip"
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
    get_wikitext2(level)
