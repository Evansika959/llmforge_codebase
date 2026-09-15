# Held-out documents

These files freeze the documents used to score supernet slices, so every score reproduces without
downloading FineWeb-Edu.

## Provenance

- Source: FineWeb-Edu, dataset `HuggingFaceFW/fineweb-edu`, subset `sample/10BT`.
- Shard: the last parquet file in sorted order, `013_00000.parquet`. Supernet training reads the
  shards in sorted order starting from the first, so this shard is unseen during training.
- Selection: rows in file order, keeping only documents longer than 500 characters.
- Truncation: every document is cut to its first 4000 characters.

## Files

| file | documents | content |
|---|---|---|
| `heldout_ab.json` | 192 | documents 0 to 191 of the filtered stream |
| `heldout_texts.json` | 96 | the first 96 documents of `heldout_ab.json` |

- A search scores documents 0 to 31. Validation re-scores documents 32 to 95, so the two never
  share a document.
- The two halves of `heldout_ab.json`, documents 0 to 95 and documents 96 to 191, are the disjoint
  sets A and B of the predictor stability study.
- Both files were checked against documents regenerated from the parquet shard and match exactly.

Load the documents through the package loader, which reads these files first and falls back to the
parquet shard only when a request goes past document 191:

```python
from llmforge.supernet.data.heldout import heldout_texts

search_docs = heldout_texts(32)
validation_docs = heldout_texts(64, skip=32)
```

## License

FineWeb-Edu is released under the Open Data Commons Attribution License v1.0, ODC-By, and its use
is also subject to the Common Crawl Terms of Use. These excerpts are redistributed under the same
terms, with attribution to the FineWeb-Edu dataset.
