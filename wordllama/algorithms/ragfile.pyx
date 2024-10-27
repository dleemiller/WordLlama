# wordllama/algorithms/ragfile.pyx

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdio cimport FILE, fopen, fread, fwrite, fclose
from libc.string cimport memcmp, memset, memcpy
from libc.time cimport time
from libc.stdlib cimport malloc, free
import uuid
import numpy as np
cimport numpy as np

# Declare the magic number array
cdef uint8_t MAGIC_NUMBER[4]

# Set the magic number manually
MAGIC_NUMBER[0] = 0x52  # 'R'
MAGIC_NUMBER[1] = 0x41  # 'A'
MAGIC_NUMBER[2] = 0x47  # 'G'
MAGIC_NUMBER[3] = 0x01  # '\x01'

cdef const uint16_t HEADER_VERSION = 1

cdef struct Header:
    uint8_t magic_number[4]
    uint16_t version
    uint16_t flags
    uint32_t vector_dim
    uint8_t file_hash[32]
    uint8_t is_binary
    uint8_t model_id_hash[32]
    uint64_t data_size
    char data_format[16]
    uint64_t timestamp
    uint8_t uuid[16]
    uint32_t header_checksum

# Function to calculate UUID
cdef void calculate_uuid(uint8_t* uuid_bytes):
    cdef uuid.UUID u = uuid.uuid4()
    memcpy(uuid_bytes, u.bytes, 16)

# Function to calculate checksum
cdef uint32_t calculate_checksum(Header* header):
    cdef uint32_t checksum = 0
    cdef uint8_t* data = <uint8_t*>header
    cdef int i
    for i in range(sizeof(Header) - 4):  # Exclude the checksum field itself
        checksum += data[i]
    return checksum

# Function to create the header
cdef Header* create_header(uint32_t vector_dim, uint8_t is_binary, char* model_id_hash, uint64_t data_size, char* data_format):
    cdef Header* header = <Header*>malloc(sizeof(Header))
    if not header:
        raise MemoryError("Failed to allocate memory for Header")
    
    memcpy(header.magic_number, MAGIC_NUMBER, 4)
    header.version = HEADER_VERSION
    header.flags = 0
    header.vector_dim = vector_dim
    memset(header.file_hash, 0, 32)  # You can later fill this with an actual hash
    header.is_binary = is_binary
    memcpy(header.model_id_hash, model_id_hash, 32)
    header.data_size = data_size
    memcpy(header.data_format, data_format, 16)
    header.timestamp = time(NULL)
    calculate_uuid(header.uuid)
    header.header_checksum = calculate_checksum(header)

    return header

# Function to write the ragfile
cdef void write_ragfile(const char* filename, Header* header, np.ndarray[np.uint8_t, ndim=1] embeddings, np.ndarray[np.uint8_t, ndim=1] binary_data):
    cdef FILE* f = fopen(filename, "wb")
    if not f:
        raise IOError("Failed to open file for writing")
    
    # Write the static header
    fwrite(header, sizeof(Header), 1, f)
    
    # Write the embeddings
    fwrite(<void*>embeddings.data, 1, embeddings.nbytes, f)

    # Calculate and write the padding
    cdef int padding_size = 4096 - (sizeof(Header) + embeddings.nbytes)
    cdef char padding[4096]
    memset(padding, 0, padding_size)
    fwrite(padding, 1, padding_size, f)

    # Write the binary data
    fwrite(<void*>binary_data.data, 1, binary_data.nbytes, f)

    fclose(f)

# Function to read the string from the ragfile
cdef char* read_string_from_ragfile(const char* filename, Header* header):
    cdef FILE* f = fopen(filename, "rb")
    if not f:
        raise IOError("Failed to open file for reading")
    
    # Skip the header and padding
    fseek(f, 4096, SEEK_SET)
    
    # Allocate space for the string
    cdef char* string_data = <char*>malloc(header.data_size)
    if not string_data:
        fclose(f)
        raise MemoryError("Failed to allocate memory for string data")

    fread(string_data, header.data_size, 1, f)
    fclose(f)
    
    return string_data

# Function to deallocate the header
cdef void deallocate_header(Header* header):
    if header:
        free(header)

