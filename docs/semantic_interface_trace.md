# Semantic interface trace

1. Raw V2 structural validation ends in `normalize_and_validate_v2_contract` / `validate_v2_live_contract`, after schema, endpoint, cardinality, and operation-shape checks.
2. Semantic typing begins when raw roles, relation phrases, and operation phrases are converted into finite runtime-role hypotheses.
3. Canonical role commitment currently occurs in `resolve_role_type_hypotheses`; before this change a successful function mapper incorrectly initialized a singleton domain.
4. Relation signatures constrain both endpoint domains through predicate-registry subject/object role pairs.
5. Operation signatures constrain source, target, and anchor domains through robot-capability participant signatures; this is independent of geometric precondition templates.
6. `TASK_SPECIFICATION_FAILURE` is produced only by raw structural errors or when no interpretation satisfies explicit FM semantic-family and endpoint-structure claims.
7. `GRAPH_COMPILATION_FAILURE` is produced after task-specification satisfiability when the runtime ontology or IR cannot completely represent the coherent FM semantics.

The validator consumes the pure hypothesis layer and checks existence of a coherent interpretation. The compiler consumes the same hypotheses to build canonical or provisional `G_F`; it owns runtime representability and canonical commitment.
