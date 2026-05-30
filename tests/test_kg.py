import tempfile
import pathlib
from kernel_factory.kg.graph import KernelFactoryKG


def test_kg_opens_and_closes():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.close()


def test_upsert_kernel_template():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.upsert_kernel_template(
            name="matmul_v1",
            op_type="matmul",
            template_str="def kernel(a_ref, b_ref, o_ref): ...",
            verified=True,
        )
        results = kg.query("MATCH (t:KernelTemplate) RETURN t.name")
        names = [r[0] for r in results]
        assert "matmul_v1" in names
        kg.close()


def test_upsert_hardware_limits():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.upsert_hardware_limits(
            tpu_version="v5e",
            vmem_bytes=128 * 1024 * 1024,
            vector_width=128,
            sublane_width=8,
        )
        results = kg.query("MATCH (h:HardwareLimits) RETURN h.tpu_version")
        versions = [r[0] for r in results]
        assert "v5e" in versions
        kg.close()


def test_get_template_returns_verified():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.upsert_kernel_template(
            name="rmsnorm_v1", op_type="rmsnorm",
            template_str="def kernel(): pass", verified=True,
        )
        result = kg.get_template("rmsnorm")
        assert result == "def kernel(): pass"
        kg.close()


def test_get_template_returns_none_for_missing():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        assert kg.get_template("attention") is None
        kg.close()
