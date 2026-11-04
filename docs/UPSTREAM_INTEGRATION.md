# Upstream integration

This release consumes already-selected evidence sentences. It exposes a one-shot
Docker CLI with a CSV interface; it does not include or invoke a classifier.

## CSV hand-off

Export the upstream selected sentences to `input_sentences.csv` with the header:

```csv
row_id,PMID,sentence
```

`row_id` must be a unique integer per sentence. Keep it stable when resuming a run.
The upstream system may choose any internal scores, labels, or model; this stage
only requires the selected sentences and their source PMID. If `pseudo_label` is
also supplied, the extractor retains only rows whose value is `POS`.

Export document context to `pmid_metadata.csv`:

```csv
PMID,Species,Tissue
```

Use `Human` / `Mouse` for supported species-specific gene retrieval, and an empty
`Tissue` value when unavailable. Multiple sentence rows can reference the same
metadata PMID. Quote sentence fields containing commas or line breaks using a
standard CSV writer. Finish writing both files before launching curation; an
atomic rename is suitable if the upstream exporter runs concurrently.

## Existing upstream output directory

In this repository's `.env`, point directly to the upstream artifacts:

```dotenv
CMAGENT_INPUT_DIR=/absolute/path/to/upstream/output
CMAGENT_OUTPUT_DIR=/absolute/path/to/curation/output
CMAGENT_RESOURCES_DIR=/absolute/path/to/reference-resources
CMAGENT_RUN_NAME=batch_001
```

Create the output directory as the host user and set its identity for the job:

```bash
mkdir -p /absolute/path/to/curation/output
export CMAGENT_UID="$(id -u)" CMAGENT_GID="$(id -g)"
docker compose build pipeline
docker compose run --rm pipeline
```

The default container command reads `/inputs/input_sentences.csv` and
`/inputs/pmid_metadata.csv`. Other filenames can be passed with `--input` and
`--metadata`, using the `/inputs/` paths visible inside the container.

## Run after another Compose service

After building this repository's image (`docker compose build pipeline`), a
curation service can be added to an existing upstream Compose project. This
example assumes that the existing upstream service is named `classifier`, exits
after producing its two files, and writes to `./shared/sentences`:

```yaml
services:
  # classifier: ... existing upstream service, writing ./shared/sentences ...
  curation:
    image: cmagent-multi-agent:local
    user: "${CMAGENT_UID:-10001}:${CMAGENT_GID:-10001}"
    depends_on:
      classifier:
        condition: service_completed_successfully
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DATABASE_URL: sqlite:////outputs/cell_marker.db
      CMAGENT_RESOURCES_DIR: /resources
      CMAGENT_RUNS_DIR: /outputs/runs
    volumes:
      - ./shared/sentences:/inputs:ro
      - ./shared/curation:/outputs
      - ./reference-resources:/resources:ro
    command:
      - --input
      - /inputs/input_sentences.csv
      - --metadata
      - /inputs/pmid_metadata.csv
      - --database-output
      - /outputs/cell_marker_database.csv
      - --run-name
      - ${CMAGENT_RUN_NAME:-batch_001}
```

Both services must be part of the same Compose project for `depends_on` to apply.
For a long-running upstream service, trigger this job only after its export is
complete instead of using `service_completed_successfully`.

## Outputs and resumption

- `runs/<run-name>/extraction/`: prepared work items, extraction/status records,
  deduplicated relations, and selective verification artifacts.
- `runs/<run-name>/mapping/`: local mappings, cell/gene decisions, and final
  `cell_gene_agent.csv`, including unresolved candidates.
- `runs/<run-name>/controller_manifest.json`: stage status and run configuration.
- `cell_marker_database.csv`: merged canonical relations with source lists.
- `cell_marker.db`: SQLite relational records and evidence by default.

The Docker output bind mount preserves artifacts after the container exits. The
pipeline returns a nonzero exit status when an invoked stage raises an error;
the verifier's documented fail-open behavior may retain candidates with a
recorded failure route. Re-run the same command for the same inputs/configuration
to resume checkpointed work. Use a fresh run name for new inputs, models, or
reference resources. A completed run with no extracted relations leaves empty
run artifacts and does not create/update canonical databases.

For PostgreSQL, put its SQLAlchemy URL in `DATABASE_URL` and make the server
reachable from the container's Docker network. No PostgreSQL service or web
frontend is required for the default SQLite workflow.
