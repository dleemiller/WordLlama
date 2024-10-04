# distutils: language=c++
# cython: language_level=3, infer_types=True, binding=True, boundscheck=False, wraparound=False
import cython
from cpython cimport PyObject, Py_INCREF, array
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libc.string cimport memchr


cdef extern from "Python.h":
    int PyObject_Length(object o) except -1

cdef struct ItemWithSize:
    PyObject* obj
    Py_ssize_t size

cdef vector[ItemWithSize] make_batch(vector[ItemWithSize]& items):
    cdef vector[ItemWithSize] result
    cdef ItemWithSize item
    for item in items:
        Py_INCREF(<object>item.obj)
        result.push_back(item)
    return result

def constrained_batches(iterable, Py_ssize_t max_size, Py_ssize_t max_count=-1, get_len=None, bint strict=True):
    if max_size <= 0:
        raise ValueError('maximum size must be greater than zero')

    cdef vector[ItemWithSize] batch
    cdef Py_ssize_t batch_size = 0
    cdef Py_ssize_t batch_count = 0
    cdef Py_ssize_t item_len
    cdef object item
    cdef bint reached_count, reached_size
    cdef ItemWithSize item_with_size

    for item in iterable:
        if get_len is None:
            item_len = PyObject_Length(item)
        else:
            item_len = get_len(item)

        if strict and item_len > max_size:
            raise ValueError('item size exceeds maximum size')

        reached_count = max_count != -1 and batch_count == max_count
        reached_size = item_len + batch_size > max_size

        if batch_count and (reached_size or reached_count):
            yield tuple((<object>item.obj for item in make_batch(batch)))
            batch.clear()
            batch_size = 0
            batch_count = 0

        item_with_size.obj = <PyObject*>item
        item_with_size.size = item_len
        batch.push_back(item_with_size)
        batch_size += item_len
        batch_count += 1

    if not batch.empty():
        yield tuple((<object>item.obj for item in make_batch(batch)))

def split_sentences(str text not None, set punct_chars=None):
    cdef Py_ssize_t i, start = 0, text_len = len(text)
    cdef list sentences = []
    cdef bint seen_period = False
    cdef Py_UCS4 current_char
    cdef cset[Py_UCS4] punct_chars_c

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
    else:
        punct_chars = set(punct_chars)

    punct_chars_c = cset[Py_UCS4](ord(c) for c in punct_chars)

    # Fast path: if no punctuation is found, return the whole text as a single sentence
    if not any(c in punct_chars for c in text):
        return [text]

    for i in range(text_len):
        current_char = ord(text[i])
        if punct_chars_c.find(current_char) != punct_chars_c.end():  # Use find() for cset membership check
            seen_period = True
        elif seen_period and (current_char == ord(' ') or current_char == ord('\n')):
            if i + 1 < text_len and (text[i+1].isupper() or text[i+1] == '\n'):
                sentences.append(text[start:i+1].strip())
                start = i + 1
            seen_period = False

    if start < text_len:
        sentences.append(text[start:].strip())

    return sentences

def constrained_coalesce(list iterable, Py_ssize_t max_size, str separator="\n", Py_ssize_t max_iterations=100):
    """
    Recursively coalesces pairs of successive items from the iterable as long as the
    combined size of two items doesn't exceed max_size. The separator used for joining
    pairs can be configured.

    Parameters:
        iterable (list): List of strings to be coalesced.
        max_size (Py_ssize_t): Maximum allowed size of the combined string.
        separator (str): The string used to join pairs of items (default: newline).
        max_iterations (Py_ssize_t): Maximum number of list iterations (default: 100).

    Returns:
        list: Coalesced list.
    """
    cdef list result = iterable
    cdef Py_ssize_t changes = 1
    cdef Py_ssize_t iteration = 0

    # Recurse the list until no further combinations can be made or max_iterations is reached
    while changes > 0 and iteration < max_iterations:
        result, changes = _combine_pass(result, max_size, separator)
        iteration += 1
    return result

cdef tuple _combine_pass(list iterable, Py_ssize_t max_size, str separator):
    """
    A single pass to combine successive pairs of items if their combined size doesn't
    exceed max_size. The separator used for joining pairs can be configured.
    
    Returns:
        tuple: A new list of coalesced items and the number of changes made.
    """
    cdef vector[PyObject*] combined
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t changes = 0
    cdef Py_ssize_t current_size, next_size, combined_size
    cdef object current_item, next_item, combined_item

    while i < len(iterable):
        current_item = iterable[i]
        current_size = len(current_item)

        if i + 1 < len(iterable):  # Check if we have a pair to combine
            next_item = iterable[i + 1]
            next_size = len(next_item)

            # Try to combine the two items
            combined_size = current_size + next_size + len(separator)
            if combined_size <= max_size:
                # Create a new combined string using the specified separator
                combined_item = separator.join([current_item, next_item])

                # Ensure reference management for Python objects
                combined.push_back(<PyObject*>combined_item)
                Py_INCREF(combined_item)
                changes += 1
                i += 2  # Move past the combined pair
            else:
                # Ensure reference management
                combined.push_back(<PyObject*>current_item)
                Py_INCREF(current_item)
                i += 1  # Only move past the first item
        else:
            # Handle the last item in case of an odd-length list
            combined.push_back(<PyObject*>current_item)
            Py_INCREF(current_item)
            i += 1

    return [<object>combined[j] for j in range(combined.size())], changes

cpdef list reverse_merge(list strings, int n, str separator="\n"):
    cdef list result = []
    cdef str current_str
    cdef int i, str_len
    cdef str string

    if len(strings) == 0:
        return result

    current_str = strings[0]  # Start at index 0 with the first string

    for i in range(1, len(strings)):  # Start at index 1
        string = strings[i]
        str_len = len(string)

        if str_len < n:
            if current_str:
                current_str = separator.join([current_str, string])
            else:
                current_str = string
        else:
            result.append(current_str)
            current_str = string

    result.append(current_str)

    return result

