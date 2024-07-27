# cython: language_level=3, infer_types=True, binding=True
import cython
from typing import List

@cython.boundscheck(False)
@cython.wraparound(False)
def split_sentences(str text, set punct_chars=None) -> List[str]:
    """
    Split text into sentences based on punctuation characters.
    
    Parameters:
        text (str): The input text to split.
        punct_chars (set): A set of punctuation characters to use for splitting.
        
    Returns:
        List[str]: A list of sentences.
    """
    cdef int i, start = 0
    cdef list sentences = []
    cdef bint seen_period = False  # Use bint for boolean type
    cdef str current_char

    if punct_chars is None:
        punct_chars = {
            '.', '!', '?', '։', '؟', '۔', '܀', '܁', '܂', '߹', '।', '॥', '၊', '။', '።', '፧', '፨', 
            '᙮', '᜵', '᜶', '᠃', '᠉', '᥄', '᥅', '᪨', '᪩', '᪪', '᪫', '᭚', '᭛', '᭞', '᭟', 
            '᰻', '᰼', '᱾', '᱿', '‼', '‽', '⁇', '⁈', '⁉', '⸮', '⸼', '꓿', '꘎', '꘏', '꛳', 
            '꛷', '꡶', '꡷', '꣎', '꣏', '꤯', '꧈', '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒', 
            '﹖', '﹗', '！', '．', '？', '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀', '𑃁', '𑅁', 
            '𑅂', '𑅃', '𑇅', '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼', '𑊩', '𑑋', 
            '𑑌', '𑗂', '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐', '𑗑', '𑗒', 
            '𑗓', '𑗔', '𑗕', '𑗖', '𑗗', '𙁁', '𙁂', '𑜼', '𑜽', '𑜾', '𑩂', '𑩃', '𪛝', 
            '𪛞', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈', '｡', '。'
        }
    
    for i in range(len(text)):
        current_char = text[i]
        if current_char in punct_chars:
            seen_period = True
        elif seen_period and (current_char == ' ' or current_char == '\n'):
            if i + 1 < len(text) and (text[i+1].isupper() or text[i+1] == '\n'):
                sentences.append(text[start:i+1].strip())
                start = i + 1
            seen_period = False

    if start < len(text):
        sentences.append(text[start:].strip())

    return sentences

