from datasets import load_dataset

def load_datasets(seed=42):
    def shuffled_load(path, *args, **kwargs):
        return load_dataset(path, *args, **kwargs).shuffle(seed)

    return {
        "train": {
            # NLI (Natural Language Inference) datasets
            "all-nli": shuffled_load("sentence-transformers/all-nli", "triplet", split="train"),
            "nli-for-simcse": shuffled_load("sentence-transformers/nli-for-simcse", "triplet", split="train"),

            # Information Retrieval datasets
            "msmarco": shuffled_load("sentence-transformers/msmarco-bm25", "triplet", split="train"),
            "mr-tydi": shuffled_load("sentence-transformers/mr-tydi", "en-triplet", split="train"),

            # Text Summarization / Compression datasets
            "compression": shuffled_load("sentence-transformers/sentence-compression", split="train"),
            "simple-wiki": shuffled_load("sentence-transformers/simple-wiki", split="train"),

            # News datasets
            "agnews": shuffled_load("sentence-transformers/agnews", split="train"),
            "ccnews": shuffled_load("sentence-transformers/ccnews", split="train"),
            "npr": shuffled_load("sentence-transformers/npr", split="train"),

            # Question Answering (QA) datasets
            "gooaq": shuffled_load("sentence-transformers/gooaq", split="train"),
            "yahoo-answers": shuffled_load("sentence-transformers/yahoo-answers", "title-question-answer-pair", split="train"),
            "eli5": shuffled_load("sentence-transformers/eli5", split="train"),
            "amazon-qa": shuffled_load("sentence-transformers/amazon-qa", split="train[0:1000000]"),
            "squad": shuffled_load("sentence-transformers/squad", split="train"),
            "natural_questions": shuffled_load("sentence-transformers/natural-questions", split="train"),
            "hotpotqa": shuffled_load("sentence-transformers/hotpotqa", "triplet", split="train"),

            # Duplicate Detection datasets
            "quora_duplicates": shuffled_load("sentence-transformers/quora-duplicates", "pair", split="train"),
            "quora_triplets": shuffled_load("sentence-transformers/quora-duplicates", "triplet", split="train"),

            # Scientific / Academic datasets
            "specter": shuffled_load("sentence-transformers/specter", "triplet", split="train"),

            # Stack Exchange datasets
            "stackexchange_bbp": shuffled_load("sentence-transformers/stackexchange-duplicates", "body-body-pair", split="train"),
            "stackexchange_ttp": shuffled_load("sentence-transformers/stackexchange-duplicates", "title-title-pair", split="train"),
            "stackexchange_ppp": shuffled_load("sentence-transformers/stackexchange-duplicates", "post-post-pair", split="train"),

            # Lexical / Linguistic datasets
            "altlex": shuffled_load("sentence-transformers/altlex", split="train"),
        },
        "eval": {
            # Evaluation datasets
            "all-nli": shuffled_load("sentence-transformers/all-nli", "triplet", split="dev"),
            "stsb": shuffled_load("sentence-transformers/stsb", split="test"),
        },
        "sts_validation": ("sentence-transformers/stsb",),
    }
