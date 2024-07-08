from datasets import load_dataset


def load_datasets():
    return {
        "train": {
            "all-nli": load_dataset(
                "sentence-transformers/all-nli", "triplet", split="train"
            ),
            "msmarco": load_dataset(
                "sentence-transformers/msmarco-bm25", "triplet", split="train"
            ),
            "msmarco2": load_dataset(
                "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
                "triplet",
                split="train",
            ),
            "hotpotqa": load_dataset(
                "sentence-transformers/hotpotqa", "triplet", split="train"
            ),
            "nli-for-simcse": load_dataset(
                "sentence-transformers/nli-for-simcse", "triplet", split="train"
            ),
            "mr-tydi": load_dataset(
                "sentence-transformers/mr-tydi", "en-triplet", split="train"
            ),
            "compression": load_dataset(
                "sentence-transformers/sentence-compression", split="train"
            ),
            "agnews": load_dataset("sentence-transformers/agnews", split="train"),
            "gooaq": load_dataset("sentence-transformers/gooaq", split="train"),
            "yahoo": load_dataset(
                "sentence-transformers/yahoo-answers",
                "title-question-answer-pair",
                split="train",
            ),
            "eli5": load_dataset("sentence-transformers/eli5", split="train"),
            "specter": load_dataset(
                "sentence-transformers/specter", "triplet", split="train"
            ),
            "quora_duplicates": load_dataset(
                "sentence-transformers/quora-duplicates", "pair", split="train"
            ),
            "amazon-qa": load_dataset(
                "sentence-transformers/amazon-qa", split="train[0:1000000]"
            ),
            "squad": load_dataset("sentence-transformers/squad", split="train"),
            "stackexchange_bbp": load_dataset(
                "sentence-transformers/stackexchange-duplicates",
                "body-body-pair",
                split="train",
            ),
            "stackexchange_ttp": load_dataset(
                "sentence-transformers/stackexchange-duplicates",
                "title-title-pair",
                split="train",
            ),
            "stackexchange_ppp": load_dataset(
                "sentence-transformers/stackexchange-duplicates",
                "post-post-pair",
                split="train",
            ),
            "quora_triplets": load_dataset(
                "sentence-transformers/quora-duplicates", "triplet", split="train"
            ),
            "natural_questions": load_dataset(
                "sentence-transformers/natural-questions", split="train"
            ),
            "altlex": load_dataset("sentence-transformers/altlex", split="train"),
        },
        "eval": {
            "all-nli": load_dataset(
                "sentence-transformers/all-nli", "triplet", split="dev"
            ),
            "stsb": load_dataset("sentence-transformers/stsb", split="test"),
        },
        "sts_validation": ("sentence-transformers/stsb",),
    }
