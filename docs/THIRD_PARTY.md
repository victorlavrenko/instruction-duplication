# Third-party data

The lexical exposure metric uses `results/yearly-counts.csv.gz` from
[`berenslab/llm-excess-vocab`](https://github.com/berenslab/llm-excess-vocab),
the reproducibility repository for Kobak et al. (2025), “Delving into
LLM-assisted writing in biomedical publications through excess vocabulary.”

- Repository commit: `53db991afc251782106cd817a1c3fa47a4d41781`
- File SHA-256: `e42e37c9ad5abc4e098e0ea02558399b8557d85332bf350942c6cb9bda9d9d93`
- Coverage: 15,103,887 PubMed abstracts, 2010–2024
- Upstream license: MIT

The source file is downloaded to a user cache and retained unchanged. Each
experiment workspace stores only the document frequencies and calculated IDF
values needed by its selected question stems, together with the source identity
above.
