"""
Export the tiktoken "gpt2" mergeable_ranks to a flat binary for the C BPE tokenizer (bpe.h).

Format (little-endian):
  int32 n          # number of byte-tokens (50256)
  int32 eot        # id of <|endoftext|> (50256)
  repeat n times, in id order 0..n-1:
    int32 len
    uint8 bytes[len]

Usage: python export_gpt2_tokenizer.py tokenizer_gpt2.bin
"""
import struct, sys
import tiktoken

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_gpt2.bin"
    enc = tiktoken.get_encoding("gpt2")
    mr = enc._mergeable_ranks               # dict[bytes, int], ids 0..50255
    eot = enc._special_tokens["<|endoftext|>"]
    n = len(mr)
    id_to_bytes = [None] * n
    for b, i in mr.items():
        id_to_bytes[i] = b
    assert all(x is not None for x in id_to_bytes), "non-contiguous ids"
    with open(out, "wb") as f:
        f.write(struct.pack("ii", n, eot))
        for b in id_to_bytes:
            f.write(struct.pack("i", len(b)))
            f.write(b)
    print(f"wrote {out}: n={n} eot={eot}")

if __name__ == "__main__":
    main()
