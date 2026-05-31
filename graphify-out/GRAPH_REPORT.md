# Graph Report - /Users/myangupta/Downloads/QuackHack  (2026-05-30)

## Corpus Check
- Corpus is ~7,201 words - fits in a single context window. You may not need a graph.

## Summary
- 159 nodes · 350 edges · 12 communities
- Extraction: 65% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 124 edges (avg confidence: 0.61)
- Token cost: 12,800 input · 2,400 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Pydantic Schema Layer|Pydantic Schema Layer]]
- [[_COMMUNITY_KG Node Type Definitions|KG Node Type Definitions]]
- [[_COMMUNITY_KernelFactoryKG Graph Operations|KernelFactoryKG Graph Operations]]
- [[_COMMUNITY_Hardware Limits & VMEM Config|Hardware Limits & VMEM Config]]
- [[_COMMUNITY_Assembler & Template Tests|Assembler & Template Tests]]
- [[_COMMUNITY_Tile Solver & Tests|Tile Solver & Tests]]
- [[_COMMUNITY_Verification Gate & Tests|Verification Gate & Tests]]

## God Nodes (most connected - your core abstractions)
1. `LayerSpec` - 24 edges
2. `KernelConfig` - 24 edges
3. `DType` - 22 edges
4. `TileSolver` - 21 edges
5. `KernelFactoryKG` - 17 edges
6. `VerificationGate` - 15 edges
7. `Assembler` - 14 edges
8. `HardwareLimits` - 13 edges
9. `TestResult` - 12 edges
10. `_config()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Kuzu knowledge graph for kernel provenance` --conceptually_related_to--> `KernelFactoryKG`  [INFERRED]
  docs/superpowers/plans/2026-05-30-tpu-kernel-factory.md → src/kernel_factory/kg/graph.py
- `Solver → RAG → Assembly → Verify pipeline` --conceptually_related_to--> `TileSolver`  [INFERRED]
  docs/superpowers/plans/2026-05-30-tpu-kernel-factory.md → src/kernel_factory/solver.py
- `Solver → RAG → Assembly → Verify pipeline` --conceptually_related_to--> `Assembler`  [INFERRED]
  docs/superpowers/plans/2026-05-30-tpu-kernel-factory.md → src/kernel_factory/assembler.py
- `LanceDB RAG template retrieval` --conceptually_related_to--> `TEMPLATES`  [INFERRED]
  docs/superpowers/plans/2026-05-30-tpu-kernel-factory.md → src/kernel_factory/templates.py
- `VerifyMode` --conceptually_related_to--> `Remote TPU VM verification`  [INFERRED]
  src/kernel_factory/verify.py → docs/superpowers/plans/2026-05-30-tpu-kernel-factory.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Solver → Assembler → VerificationGate kernel generation pipeline** — solver_tilesolver, assembler_assembler, verify_verificationgate, schemas_layerspec, schemas_kernelconfig [INFERRED 0.95]
- **KernelFactoryKG schema-driven node and edge DDL** — kg_schema_node_schemas, kg_schema_edge_schemas, kg_graph_kernelfactorykg [EXTRACTED 1.00]
- **Pydantic schema layer shared across solver, assembler, and verifier** — schemas_layerspec, schemas_hardwarelimits, schemas_kernelconfig, schemas_testresult [INFERRED 0.95]

## Communities (12 total, 0 thin omitted)

### Community 0 - "Pydantic Schema Layer"
Cohesion: 0.18
Nodes (28): BaseModel, Connection, DType, Enum, float, DType, KernelConfig, LayerSpec (+20 more)

### Community 1 - "KG Node Type Definitions"
Cohesion: 0.10
Nodes (32): Assembler, BenchmarkResult (KG node type), CompileResult (KG node type), GeneratedKernel (KG node type), HardwareLimits (KG node type), KernelTemplate (KG node type), NODE_SCHEMAS, Kuzu knowledge graph for kernel provenance (+24 more)

### Community 2 - "KernelFactoryKG Graph Operations"
Cohesion: 0.14
Nodes (12): KernelFactoryKG, EDGE_SCHEMAS, bool, int, Path, str, KernelFactoryKG tests, test_get_template_returns_none_for_missing() (+4 more)

### Community 3 - "Hardware Limits & VMEM Config"
Cohesion: 0.18
Nodes (10): HardwareLimits, Usable VMEM after safety margin (default 75% of total)., _aligned(), _vmem_matmul(), _vmem_rmsnorm(), int, bool, int (+2 more)

### Community 4 - "Assembler & Template Tests"
Cohesion: 0.30
Nodes (13): Assembler, _config(), KernelConfig, Assembled code must be parseable by Python's AST., test_assemble_matmul_has_run_function(), test_assemble_matmul_injects_block_sizes(), test_assemble_matmul_injects_dtypes(), test_assemble_matmul_num_k_tiles_correct() (+5 more)

### Community 5 - "Tile Solver & Tests"
Cohesion: 0.35
Nodes (11): TileSolver, _hw(), HardwareLimits, test_solver_last_dim_aligned_to_vector_width(), test_solver_raises_for_unsupported_op(), test_solver_returns_config_for_matmul(), test_solver_rmsnorm(), test_solver_second_dim_aligned_to_sublane() (+3 more)

### Community 6 - "Verification Gate & Tests"
Cohesion: 0.38
Nodes (11): VerificationGate, Verification harness tests — CPU interpret=True mode only, no TPU required., _solve(), test_logged_to_sqlite(), test_matmul_cpu_interpret_passes(), test_multiple_runs_accumulate_in_db(), test_result_has_latency(), test_result_has_max_abs_error() (+3 more)

## Knowledge Gaps
- **11 isolated node(s):** `Path`, `bool`, `int`, `MATMUL_TEMPLATE`, `RMSNORM_TEMPLATE` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TileSolver` connect `Tile Solver & Tests` to `Pydantic Schema Layer`, `Hardware Limits & VMEM Config`, `Verification Gate & Tests`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `LayerSpec` connect `Pydantic Schema Layer` to `Hardware Limits & VMEM Config`, `Assembler & Template Tests`, `Tile Solver & Tests`, `Verification Gate & Tests`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `KernelConfig` connect `Pydantic Schema Layer` to `Hardware Limits & VMEM Config`, `Assembler & Template Tests`, `Tile Solver & Tests`, `Verification Gate & Tests`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `LayerSpec` (e.g. with `Connection` and `DType`) actually correct?**
  _`LayerSpec` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `KernelConfig` (e.g. with `Connection` and `DType`) actually correct?**
  _`KernelConfig` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `DType` (e.g. with `Connection` and `DType`) actually correct?**
  _`DType` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `TileSolver` (e.g. with `DType` and `HardwareLimits`) actually correct?**
  _`TileSolver` has 15 INFERRED edges - model-reasoned connections that need verification._