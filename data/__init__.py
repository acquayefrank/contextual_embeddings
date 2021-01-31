from pathlib import Path

DATA_ROOT = Path(__file__).parent

TRAIN_DATA_PATH = DATA_ROOT / "train.csv"
WORDS_DATA_PATH = DATA_ROOT / "_words_with_hyponyms.txt"
WORDS_FROM_EMBEDDINGS_DATA_PATH = (
    DATA_ROOT / "_words_from_word_embeddings_with_hyponyms.txt"
)
COMMON_WORDS_FINAL_DATA = DATA_ROOT / "common_words_100_final_data.csv"
