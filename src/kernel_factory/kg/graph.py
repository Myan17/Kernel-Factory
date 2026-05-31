from __future__ import annotations
import pathlib
import kuzu
from kernel_factory.kg.schema import NODE_SCHEMAS, EDGE_SCHEMAS


class KernelFactoryKG:
    def __init__(self, db_path: pathlib.Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self):
        for stmt in NODE_SCHEMAS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt + ";")
        for stmt in EDGE_SCHEMAS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt + ";")

    def query(self, cypher: str) -> list:
        result = self._conn.execute(cypher)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def upsert_kernel_template(
        self, name: str, op_type: str, template_str: str, verified: bool
    ):
        self._conn.execute(
            "MERGE (t:KernelTemplate {name: $name}) "
            "SET t.op_type = $op_type, t.template_str = $tpl, t.verified = $v",
            {"name": name, "op_type": op_type, "tpl": template_str, "v": verified},
        )

    def upsert_generated_kernel(self, kernel_id: str, code: str, created_at: str):
        self._conn.execute(
            "MERGE (g:GeneratedKernel {kernel_id: $id}) "
            "SET g.code = $code, g.created_at = $ts",
            {"id": kernel_id, "code": code, "ts": created_at},
        )

    def upsert_hardware_limits(
        self, tpu_version: str, vmem_bytes: int, vector_width: int, sublane_width: int
    ):
        self._conn.execute(
            "MERGE (h:HardwareLimits {tpu_version: $ver}) "
            "SET h.vmem_bytes = $vm, h.vector_width = $vw, h.sublane_width = $sw",
            {
                "ver": tpu_version,
                "vm": vmem_bytes,
                "vw": vector_width,
                "sw": sublane_width,
            },
        )

    def get_template(self, op_type: str) -> str | None:
        rows = self.query(
            f"MATCH (t:KernelTemplate) WHERE t.op_type = '{op_type}' AND t.verified = true "
            "RETURN t.template_str LIMIT 1"
        )
        return rows[0][0] if rows else None

    def close(self):
        del self._conn
        del self._db
