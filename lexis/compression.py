"""Integer and postings compression primitives for Lexis experiments."""

from __future__ import annotations


def variable_byte_encode(numbers: list[int]) -> bytes:
    """Encode positive integers with variable-byte coding."""
    output = bytearray()
    for number in numbers:
        if number < 1:
            raise ValueError("variable-byte values must be positive")
        chunks = [number % 128]
        number //= 128
        while number:
            chunks.append(number % 128)
            number //= 128
        ordered = list(reversed(chunks))
        for chunk in ordered[:-1]:
            output.append(chunk)
        output.append(ordered[-1] + 128)
    return bytes(output)


def variable_byte_decode(data: bytes) -> list[int]:
    numbers: list[int] = []
    value = 0
    for byte in data:
        if byte < 128:
            value = value * 128 + byte
        else:
            numbers.append(value * 128 + byte - 128)
            value = 0
    if value:
        raise ValueError("truncated variable-byte stream")
    return numbers


def gamma_encode(number: int) -> str:
    """Return Elias gamma coding for one positive integer."""
    if number < 1:
        raise ValueError("gamma values must be positive")
    binary = f"{number:b}"
    return "0" * (len(binary) - 1) + binary


def gamma_encode_all(numbers: list[int]) -> str:
    return "".join(gamma_encode(number) for number in numbers)


def gap_encode(doc_ids: list[int]) -> list[int]:
    previous = 0
    gaps: list[int] = []
    for doc_id in doc_ids:
        if doc_id <= previous:
            raise ValueError("document IDs must be strictly increasing")
        gaps.append(doc_id - previous)
        previous = doc_id
    return gaps


def compression_sizes(doc_ids: list[int]) -> dict[str, int]:
    """Return bit/byte sizes for common postings representations."""
    gaps = gap_encode(doc_ids)
    return {
        "uncompressed_bytes": len(doc_ids) * 4,
        "gaps_bytes": len(gaps) * 4,
        "variable_byte_bytes": len(variable_byte_encode(gaps)),
        "gamma_bits": len(gamma_encode_all(gaps)),
    }
