# External reference resources

Reference dictionaries and ontology/gene cache snapshots are runtime dependencies,
not classifier models. This source release does not redistribute the original
reference data, training data, or experiment-derived cache snapshots.

Set `CMAGENT_RESOURCES_DIR` to a directory with the following layout. Docker Compose
mounts it at `/resources`; local execution defaults to `<repository>/resources`.

```text
resources/
├── marker_dictionary/
│   ├── gene_collect.csv
│   └── cell_ontology_cellname.txt
├── local_entity_cache/
│   ├── gene_exact_terms.csv
│   └── cell_terms.csv
└── local_entity_cache_v3/
    ├── cell_terms.csv
    ├── cell_candidate_terms.csv
    ├── gene_terms.csv
    ├── gene_exact_terms.csv
    └── tissue_terms.csv
```

## Dictionary hints

The lexical dictionaries originate from MarkerGeneBERT's
`Database/Extra_named_entity_database` resource directory. Only the two data files
are needed; its classifier implementation and model weights are not used.

| File | Required format |
| --- | --- |
| `marker_dictionary/gene_collect.csv` | CSV with a `gene_collect` column |
| `marker_dictionary/cell_ontology_cellname.txt` | Headerless tab-separated file; the second column contains cell names |
| `local_entity_cache/gene_exact_terms.csv` | CSV with an `exact_term` column |
| `local_entity_cache/cell_terms.csv` | CSV with a `canonical_term` column |

These resources provide lexical hints only. They do not establish a relation or
supply evidence for an extracted pair. `--dictionary-hints disabled` skips this
hint preparation when the dictionaries are unavailable; the mapping cache below
is still required for mapping and normalization.

## Mapping and normalization

The v3 cache is a prepared index over Cell Ontology, NCBI Gene, tissue, and curated
entity terms. It must contain actual reference entries, not only CSV headers.

| File | Required columns |
| --- | --- |
| `cell_terms.csv` | `normalized_term,identifier,canonical_term,mapping_source` |
| `cell_candidate_terms.csv` | `normalized_term,identifier,canonical_term,mapping_source` |
| `gene_terms.csv` | `species,normalized_term,identifier,canonical_term,mapping_source` |
| `gene_exact_terms.csv` | `species,exact_term,identifier,canonical_term,mapping_source` |
| `tissue_terms.csv` | `normalized_term,identifier,canonical_term,mapping_source` |

`cell_candidate_terms.csv` is recommended; the cell retriever falls back to
`cell_terms.csv` if it is absent. CL and UBERON identifiers use underscore notation
(for example `CL_0000084`); NCBI Gene identifiers are numeric strings. Gene cache
species keys must match the supported `Human` / `Mouse` metadata values. Use the
normalization helpers in `src/mapping/ontology_mapper.py` when preparing lookup
keys. The gene symbol exact cache preserves case-sensitive keys.

`data/mapping/cell_normalization_aliases.json` and
`data/mapping/gene_normalization_aliases.json` are small runtime retrieval
configuration files and are included. They are not ground-truth labels.

The included cache builder can rebuild cell entries from an existing source cache
and a Cell Ontology OBO file while preserving the source gene/tissue tables:

```bash
python src/mapping/rebuild_cell_cache.py \
  --source-cache /path/to/source-cache \
  --ontology /path/to/cl.obo \
  --output-cache resources/local_entity_cache_v3
```

It prioritizes ontology labels and unambiguous exact synonyms for deterministic
mapping; broader, narrower, and related synonyms remain candidate-only. It does
not download or construct the gene/tissue source tables. Obtain or prepare those
separately and record resource versions when running the pipeline.
