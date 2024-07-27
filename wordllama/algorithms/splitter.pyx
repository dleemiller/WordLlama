# cython: language_level=3, infer_types=True, binding=True
import cython
from typing import List

@cython.boundscheck(False)
@cython.wraparound(False)
def split_sentences(str text, set punct_chars=None) -> List[str]:
    cdef int i, start = 0, text_len = len(text)
    cdef list sentences = []
    cdef bint seen_period = False
    cdef str current_char
    cdef set punct_chars_c

    if punct_chars is None:
        punct_chars = {'.', '!', '?', '։', '؟', '۔', '܀', '܁', '܂', '߹', '।', '॥', '၊', '။', '።', '፧', '፨', 
                       '᙮', '᜵', '᜶', '᠃', '᠉', '᥄', '᥅', '᪨', '᪩', '᪪', '᪫', '᭚', '᭛', '᭞', '᭟',
                       '᰻', '᰼', '᱾', '᱿', '‼', '‽', '⁇', '⁈', '⁉', '⸮', '⸼', '꓿', '꘎', '꘏', '꛳',
                       '꛷', '꡶', '꡷', '꣎', '꣏', '꤯', '꧈', '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒',
                       '﹖', '﹗', '！', '．', '？', '੖', '੗', '၇', '၈', 'Ⴞ', 'Ⴟ', 'Ⴠ', 'Ⴡ', 'ᅁ',
                       'ᅂ', 'ᅃ', 'ᇅ', 'ᇆ', 'ᇍ', 'ᇞ', 'ᇟ', 'ሸ', 'ሹ', 'ሻ', 'ሼ', 'ኩ', 'ᑋ',
                       'ᑌ', 'ᗂ', 'ᗃ', 'ᗉ', 'ᗊ', 'ᗋ', 'ᗌ', 'ᗍ', 'ᗎ', 'ᗏ', 'ᗐ', 'ᗑ', 'ᗒ',
                       'ᗓ', 'ᗔ', 'ᗕ', 'ᗖ', 'ᗗ', '遁', '遂', '᜼', '᜽', '᜾', 'ᩂ', 'ᩃ', 'ꛝ',
                       'ꛞ', '᱁', '᱂', '橮', '橯', '櫵', '欷', '欸', '歄', '벟', '?', '｡', '。'}

    punct_chars_c = set(ord(c) for c in punct_chars)

    if not any(ord(char) in punct_chars_c for char in text):
        return [text]

    for i in range(text_len):
        current_char = text[i]
        if ord(current_char) in punct_chars_c:
            seen_period = True
        elif seen_period and (current_char == ' ' or current_char == '\n'):
            if i + 1 < text_len and (text[i+1].isupper() or text[i+1] == '\n'):
                sentences.append(text[start:i+1].strip())
                start = i + 1
            seen_period = False

    if start < text_len:
        sentences.append(text[start:].strip())

    return sentences
