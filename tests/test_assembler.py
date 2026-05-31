import pytest
from kernel_factory.schemas import LayerSpec, DType, KernelConfig
from kernel_factory.assembler import Assembler


def _config(bm=128, bn=128, bk=128) -> KernelConfig:
    return KernelConfig(
        block_m=bm, block_n=bn, block_k=bk, stages=1,
        input_dtype=DType.BFLOAT16,
        output_dtype=DType.BFLOAT16,
        accumulator_dtype=DType.FLOAT32,
        total_vmem_estimate_bytes=2 * 1024 * 1024,
        vmem_utilization_fraction=0.015,
    )


def test_assemble_matmul_has_run_function():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = Assembler().assemble(spec, _config())
    assert "run_matmul" in code
    assert "matmul_kernel" in code


def test_assemble_matmul_injects_block_sizes():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = Assembler().assemble(spec, _config(bm=64, bn=128, bk=64))
    assert "block_m, block_n, block_k = 64, 128, 64" in code


def test_assemble_matmul_num_k_tiles_correct():
    # K=512, block_k=128 → num_k_tiles=4
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = Assembler().assemble(spec, _config(bk=128))
    assert "num_k_tiles = 4" in code


def test_assemble_matmul_injects_dtypes():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = Assembler().assemble(spec, _config())
    assert "float32" in code   # accumulator
    assert "bfloat16" in code  # output


def test_assemble_rmsnorm_has_run_function():
    spec = LayerSpec(op_type="rmsnorm", M=512, N=4096, K=4096)
    code = Assembler().assemble(spec, _config())
    assert "run_rmsnorm" in code
    assert "rmsnorm_kernel" in code


def test_assemble_rmsnorm_injects_block_sizes():
    spec = LayerSpec(op_type="rmsnorm", M=512, N=4096, K=4096)
    code = Assembler().assemble(spec, _config(bm=32, bn=256, bk=256))
    assert "block_m, block_n = 32, 256" in code


def test_assemble_raises_for_unknown_op():
    spec = LayerSpec(op_type="attention", M=512, N=512, K=512)
    with pytest.raises(ValueError, match="No template"):
        Assembler().assemble(spec, _config())


def test_assemble_output_is_valid_python():
    """Assembled code must be parseable by Python's AST."""
    import ast
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = Assembler().assemble(spec, _config())
    ast.parse(code)  # raises SyntaxError if invalid


def test_assemble_rmsnorm_output_is_valid_python():
    import ast
    spec = LayerSpec(op_type="rmsnorm", M=512, N=512, K=512)
    code = Assembler().assemble(spec, _config())
    ast.parse(code)


def test_assemble_fused_matmul_rmsnorm_has_run_function():
    spec = LayerSpec(op_type="fused_matmul_rmsnorm", M=8192, N=768, K=768)
    code = Assembler().assemble(spec, _config(bm=128, bn=768, bk=256))
    assert "run_fused_matmul_rmsnorm" in code
    assert "fused_matmul_rmsnorm_kernel" in code


def test_assemble_fused_rmsnorm_valid_python():
    import ast
    spec = LayerSpec(op_type="fused_matmul_rmsnorm", M=8192, N=768, K=768)
    code = Assembler().assemble(spec, _config(bm=128, bn=768, bk=256))
    ast.parse(code)


def test_assemble_flash_attention_has_run_function():
    spec = LayerSpec(
        op_type="flash_attention", M=2048, N=2048, K=64,
        seq_len=2048, num_heads=12, head_dim=64,
    )
    code = Assembler().assemble(spec, _config(bm=128, bn=2048, bk=128))
    assert "run_flash_attention" in code
    assert "flash_attention_kernel" in code


def test_assemble_flash_attention_valid_python():
    import ast
    spec = LayerSpec(
        op_type="flash_attention", M=2048, N=2048, K=64,
        seq_len=2048, num_heads=12, head_dim=64,
    )
    code = Assembler().assemble(spec, _config(bm=128, bn=2048, bk=128))
    ast.parse(code)


def test_assemble_flash_attention_injects_scale():
    spec = LayerSpec(
        op_type="flash_attention", M=2048, N=2048, K=64,
        seq_len=2048, num_heads=12, head_dim=64,
    )
    code = Assembler().assemble(spec, _config(bm=128, bn=2048, bk=128))
    # scale = head_dim^-0.5 = 64^-0.5 = 0.125
    assert "0.125" in code
