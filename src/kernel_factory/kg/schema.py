NODE_SCHEMAS = """
CREATE NODE TABLE IF NOT EXISTS KernelTemplate(
    name STRING,
    op_type STRING,
    template_str STRING,
    verified BOOLEAN,
    PRIMARY KEY(name)
);
CREATE NODE TABLE IF NOT EXISTS KernelSpec(
    spec_id STRING,
    op_type STRING,
    M INT64,
    N INT64,
    K INT64,
    input_dtype STRING,
    output_dtype STRING,
    accumulator_dtype STRING,
    PRIMARY KEY(spec_id)
);
CREATE NODE TABLE IF NOT EXISTS HardwareLimits(
    tpu_version STRING,
    vmem_bytes INT64,
    vector_width INT64,
    sublane_width INT64,
    PRIMARY KEY(tpu_version)
);
CREATE NODE TABLE IF NOT EXISTS TileSpec(
    tile_id STRING,
    block_m INT64,
    block_n INT64,
    block_k INT64,
    stages INT64,
    vmem_estimate_bytes INT64,
    PRIMARY KEY(tile_id)
);
CREATE NODE TABLE IF NOT EXISTS GeneratedKernel(
    kernel_id STRING,
    code STRING,
    created_at STRING,
    PRIMARY KEY(kernel_id)
);
CREATE NODE TABLE IF NOT EXISTS TestCase(
    test_id STRING,
    description STRING,
    PRIMARY KEY(test_id)
);
CREATE NODE TABLE IF NOT EXISTS CompileResult(
    result_id STRING,
    passed BOOLEAN,
    compile_time_ms DOUBLE,
    error_trace STRING,
    PRIMARY KEY(result_id)
);
CREATE NODE TABLE IF NOT EXISTS BenchmarkResult(
    bench_id STRING,
    latency_ms DOUBLE,
    tflops DOUBLE,
    PRIMARY KEY(bench_id)
);
CREATE NODE TABLE IF NOT EXISTS FailureCase(
    failure_id STRING,
    description STRING,
    error_pattern STRING,
    PRIMARY KEY(failure_id)
);
CREATE NODE TABLE IF NOT EXISTS KnownBug(
    bug_id STRING,
    description STRING,
    tpu_version STRING,
    PRIMARY KEY(bug_id)
);
CREATE NODE TABLE IF NOT EXISTS FixPattern(
    fix_id STRING,
    description STRING,
    patch_hint STRING,
    PRIMARY KEY(fix_id)
);
"""

EDGE_SCHEMAS = """
CREATE REL TABLE IF NOT EXISTS USES_TEMPLATE(FROM KernelSpec TO KernelTemplate);
CREATE REL TABLE IF NOT EXISTS CONSTRAINED_BY(FROM KernelSpec TO HardwareLimits);
CREATE REL TABLE IF NOT EXISTS HAS_TILE(FROM KernelSpec TO TileSpec);
CREATE REL TABLE IF NOT EXISTS GENERATED_FROM(FROM GeneratedKernel TO KernelSpec);
CREATE REL TABLE IF NOT EXISTS HAS_COMPILE_RESULT(FROM GeneratedKernel TO CompileResult);
CREATE REL TABLE IF NOT EXISTS HAS_BENCHMARK(FROM GeneratedKernel TO BenchmarkResult);
CREATE REL TABLE IF NOT EXISTS CAUSED_FAILURE(FROM CompileResult TO FailureCase);
CREATE REL TABLE IF NOT EXISTS KNOWN_BUG_FOR(FROM FailureCase TO KnownBug);
CREATE REL TABLE IF NOT EXISTS FIXED_BY(FROM KnownBug TO FixPattern);
CREATE REL TABLE IF NOT EXISTS VALIDATES(FROM TestCase TO GeneratedKernel);
"""
