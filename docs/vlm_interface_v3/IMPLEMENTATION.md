# Open functional-graph interface development

Base: branch `vlm-testing-pipeline`, commit `3096d86f00d6e1149e0313270a8c2e217039f6bd`.

The v2 archive contains 209 files: 140 PNG images, 32 result.json, 32
run_manifest.json, evaluation_records.json, evaluation_summary.json, one CSV,
and two Markdown tables. All 32 vlm_inputs directories contain images only.
No raw FM JSON bodies are available, so **no 32-response deterministic replay
is claimed**. The user authorized synthetic-fixture development and a fresh
frozen v3 matrix despite that archival limitation. Older three-response
FM diagnostics were not substituted for the missing 32 responses.

## Implementation boundaries

- `structural_sanitizer.py`: immutable-input schema recovery, explicit repair
  provenance, safe singular count defaults, dangling-edge/group disabling.
- `semantic_compiler.py`: typed domain mapping, canonical/soft/context/unresolved
  classification, conservative duplicate merging, full/partial acceptance.
- `executability.py`: separate pre/post-grounding requirement statuses.
- `grounding.py`: largest verified role subset after search exhaustion; original
  cardinalities remain unchanged; retained edges and groups use the same verifier.
- Domain adapters: preserve verified operation pairing, block operations with
  missing expressed participants or relations, allow independent valid candidates.
- A single A* retains its best partial path. Its problem is serialized; offline
  replay resolves the saved actions and checks preconditions independently.
- Request/A* counters record actual entry into those stages. No implicit request
  retries: a lost response must not create a second semantic model request.
- FM output is archived per variant before schema/compiler validation.

## Generic normalization rules

| Raw pattern | Generic rule | Runtime interpretation | Why no GT is needed |
|---|---|---|---|
| Supplies/ingredient/raw-material role represented by a container | Causal source function outranks physical container form | Existing material-source type | Source/destination distinction holds for any transfer task |
| Receives prepared payload | Preserve receiving participant separately from source | Existing receptacle type | Uses expressed causal function, never expected objects |
| Operation tool position + implement language + supported category evidence | Reusable instrument is distinct from installed component | Existing tool capability | Typed operation position and ontology category jointly support mapping |
| Region + support/setting + adjacency + individual/shared binding | Spatial context disambiguates support purpose | Personal/shared support | Based on expressed relations and binding, independent of variant |
| Equivalent functions, descriptions, types, counts, binding; no distinctness edge | Conservatively merge equivalent declarations | One canonical role with raw-ID provenance | Does not collapse distinct causal participants |
| Unknown descriptive property | Keep non-executable evidence | Soft evidence | Open strings need not be runtime predicate names |
| Explicit safety/mandatory property without checker | Block affected role | Unverifiable required property | Cannot claim verified execution without required evidence |
| Unsupported relation among usable relations | Preserve each phrase independently | Known relations retained, operation unresolved | Lists are legal; unsupported requirements never become assumed true |
| Non-manipulated fixed target or context region | Preserve outside selectable runtime graph | Context-only role in graph metadata | Context does not imply a manipulation operator |

Unknown semantics do not become GT roles. Detector acceptance remains controlled
by the shared system ontology; category phrases can supply secondary role evidence.
Legacy direct adapter entry points remain for compatibility; the production
`VLMSpecProvider.provide` uses the explicit three-stage interface.

## Frozen evaluation definitions

Kitchen has four user-semantic goals (two prepared/stirred/served coffees and two
served soups with distinct suitable utensils). Living Room has three (two nearby
personal settings and accessible shared entertainment control). Workshop has three
(compatible component installed, connection completed, reusable implement safely
returned). Candidate-goal satisfaction alone never proves these full-task clauses.
Goal coverage is averaged across the 20 feasible variants. Feasible success uses
20; recovery uses 13; false completion uses 12. Zero stage eligibility is N/A.
Raw semantic precision/recall/F1 uses deterministic ontology matching on the raw
response independently of compiler acceptance, not human semantic annotation.
No offline scoring result is used to repair, ground, or plan the online graph.

The task variants and their prior failure descriptions informed interface
development; these are **not untouched held-out compiler-generalization tests**.
The live matrix, if run, uses one fresh multimodal request per variant under one
frozen code/prompt SHA, without test-time adaptation.

Inference: temperature 0.0, top_p 1.0, thinking disabled. The checked-in adapter
already defaulted to 8192 tokens at the starting commit, despite the old report's
4096 claim; retain 8192 because complete schema output must fit and truncation of
missing v2 bodies cannot be audited. JSON-schema response_format was already used
by the adapter; no server restart or reconfiguration is required or authorized.
