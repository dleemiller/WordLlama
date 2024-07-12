from pathlib import Path
from tokenizers import Tokenizer


def tokenizer_from_file(file_name: str) -> Tokenizer:
    """
    Load a tokenizer from a JSON file in the tokenizers directory.

    Args:
    file_name (str): The filename of the tokenizer configuration JSON.

    Returns:
    Tokenizer: An instance of Tokenizer initialized from the given file.
    """
    # Define the path to the tokenizers directory relative to this file
    current_dir = Path(__file__).parent
    tokenizer_path = current_dir / file_name

    # Assert that the tokenizer configuration file exists
    assert (
        tokenizer_path.exists()
    ), f"Tokenizer configuration file {tokenizer_path} does not exist."

    # Load the tokenizer from the specified file
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer
