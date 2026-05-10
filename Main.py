"""
Bloom Filter — Duplicate Detection in a String Stream
======================================================
Uses a bit array + multiple hash functions to estimate whether
each incoming string has been seen before, then compares against
an exact set to measure false positives.
"""

import hashlib
import math


# ─────────────────────────────────────────────
#  Bloom Filter
# ─────────────────────────────────────────────

class BloomFilter:
    """
    Parameters
    ----------
    capacity : int   – expected number of unique items
    error_rate: float – desired false-positive probability (e.g. 0.01 = 1 %)
    """

    def __init__(self, capacity: int = 100, error_rate: float = 0.01):
        self.capacity   = capacity
        self.error_rate = error_rate

        # Optimal bit-array size  m = -n * ln(p) / (ln 2)²
        self.m = math.ceil(
            -capacity * math.log(error_rate) / (math.log(2) ** 2)
        )
        # Optimal number of hash functions  k = (m/n) * ln 2
        self.k = math.ceil((self.m / capacity) * math.log(2))

        self.bit_array = bytearray(math.ceil(self.m / 8))   # packed bits
        self.items_added = 0

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_bit(self, index: int) -> bool:
        byte_index, bit_offset = divmod(index, 8)
        return bool(self.bit_array[byte_index] & (1 << bit_offset))

    def _set_bit(self, index: int) -> None:
        byte_index, bit_offset = divmod(index, 8)
        self.bit_array[byte_index] |= (1 << bit_offset)

    def _hash_positions(self, item: str) -> list[int]:
        """
        Generate k independent bit positions for *item* using double hashing:
            pos_i = (h1 + i * h2) mod m
        h1 and h2 come from the first and second halves of SHA-256.
        """
        digest = hashlib.sha256(item.encode()).digest()
        h1 = int.from_bytes(digest[:16],  "big") % self.m
        h2 = int.from_bytes(digest[16:],  "big") % self.m
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    # ── public API ────────────────────────────────────────────────────────

    def add(self, item: str) -> None:
        """Insert *item* into the filter."""
        for pos in self._hash_positions(item):
            self._set_bit(pos)
        self.items_added += 1

    def check(self, item: str) -> bool:
        """
        Return True  → 'possibly seen before'  (may be a false positive)
        Return False → 'definitely new'         (guaranteed)
        """
        return all(self._get_bit(pos) for pos in self._hash_positions(item))

    @property
    def bits_set(self) -> int:
        return sum(bin(b).count("1") for b in self.bit_array)

    @property
    def fill_ratio(self) -> float:
        return self.bits_set / self.m

    @property
    def estimated_fp_rate(self) -> float:
        """Current theoretical false-positive probability."""
        return (1 - math.exp(-self.k * self.items_added / self.m)) ** self.k

    def __repr__(self) -> str:
        return (
            f"BloomFilter(m={self.m} bits, k={self.k} hashes, "
            f"capacity={self.capacity}, target_fp={self.error_rate:.1%})"
        )


# ─────────────────────────────────────────────
#  Stream Processor
# ─────────────────────────────────────────────

def process_stream(stream: list[str], bf: BloomFilter) -> list[dict]:
    exact_seen: set[str] = set()
    results: list[dict] = []

    for item in stream:
        bloom_says_seen = bf.check(item)
        truly_seen      = item in exact_seen

        if bloom_says_seen:
            bloom_verdict = "possibly seen before"
        else:
            bloom_verdict = "definitely new"

        if truly_seen:
            true_verdict = "duplicate"
        else:
            true_verdict = "new"

        is_false_positive = bloom_says_seen and not truly_seen

        results.append({
            "item":              item,
            "bloom_verdict":     bloom_verdict,
            "true_verdict":      true_verdict,
            "is_false_positive": is_false_positive,
        })

        # Update both trackers AFTER checking
        bf.add(item)
        exact_seen.add(item)

    return results


# ─────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────

def print_report(results: list[dict], bf: BloomFilter) -> None:
    col_w = max(len(r["item"]) for r in results) + 2

    header = (
        f"{'#':<4} {'Item':<{col_w}} {'Bloom verdict':<26} "
        f"{'Truth':<12} {'FP?'}"
    )
    sep = "─" * len(header)

    print("\n" + "═" * len(header))
    print("  BLOOM FILTER — DUPLICATE DETECTION RESULTS")
    print("═" * len(header))
    print(header)
    print(sep)

    false_positives = 0
    true_duplicates = 0

    for i, r in enumerate(results, 1):
        fp_marker = "  ← FALSE POSITIVE" if r["is_false_positive"] else ""
        if r["is_false_positive"]:
            false_positives += 1
        if r["true_verdict"] == "duplicate":
            true_duplicates += 1

        print(
            f"{i:<4} {r['item']:<{col_w}} {r['bloom_verdict']:<26} "
            f"{r['true_verdict']:<12}{fp_marker}"
        )

    print(sep)

    total        = len(results)
    new_items    = total - true_duplicates
    fp_rate      = false_positives / new_items if new_items else 0.0

    print(f"\n  Filter   : {bf}")
    print(f"  Stream   : {total} items total")
    print(f"  Unique   : {new_items}  |  True duplicates : {true_duplicates}")
    print(f"  Bits set : {bf.bits_set} / {bf.m}  ({bf.fill_ratio:.1%} full)")
    print(f"\n  False positives    : {false_positives}")
    print(f"  Observed FP rate   : {fp_rate:.1%}  ({false_positives}/{new_items} new items misclassified)")
    print(f"  Theoretical FP rate: {bf.estimated_fp_rate:.4%}")
    print("═" * len(header) + "\n")


# ─────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────

def main() -> None:
    stream = [
        # usernames
        "alice", "bob", "carol", "alice", "dave",
        "eve", "frank", "bob", "grace", "hank",
        # URLs
        "https://example.com/home", "https://foo.bar/api",
        "https://example.com/home",           # duplicate URL
        # document titles
        "Q1_report_2024", "Q2_report_2024",
        "invoice_#4521", "Q1_report_2024",    # duplicate doc
        # more items
        "ivan", "judy", "https://foo.bar/api", # another duplicate
    ]

    print(f"\nStream ({len(stream)} items): {stream}\n")

    bf = BloomFilter(capacity=50, error_rate=0.05)
    print(f"Initialised: {bf}\n")

    results = process_stream(stream, bf)
    print_report(results, bf)


if __name__ == "__main__":
    main()