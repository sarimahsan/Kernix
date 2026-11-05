"""Unit tests for the PIR Schema."""

import unittest
from gpuperf.schema.pir import (
    PIRReport,
    HardwareInfo,
    LaunchConfig,
    PerformanceMetrics,
    WarpStallReason,
    BottleneckDiagnosis
)


class TestPIRSchema(unittest.TestCase):
    def test_pir_creation_and_serialization(self):
        hw = HardwareInfo(device="NVIDIA L4", arch="sm_89", num_sms=58)
        launch = LaunchConfig(
            grid=[64, 1, 1],
            block=[128, 1, 1],
            registers_per_thread=64,
            shared_mem_per_block_kb=16.0,
            spill_loads_bytes=0,
            spill_stores_bytes=0
        )
        perf = PerformanceMetrics(
            latency_us=120.5,
            sol_compute_pct=45.0,
            sol_memory_dram_pct=72.0,
            achieved_occupancy_pct=48.0,
            theoretical_occupancy_pct=62.5,
            occupancy_limiter="Registers",
            warp_stall_reasons=[WarpStallReason(reason="Long Scoreboard", pct=45.0)]
        )
        diag = BottleneckDiagnosis(
            primary_bottleneck="MemoryBandwidthBound",
            confidence=0.92,
            explanation="Kernel is memory bound."
        )

        pir = PIRReport(
            kernel_name="test_matmul",
            language="triton",
            hardware=hw,
            launch_config=launch,
            performance_metrics=perf,
            diagnosis=diag
        )

        # Test dictionary serialization
        data = pir.to_dict()
        self.assertEqual(data["kernel_name"], "test_matmul")
        self.assertEqual(data["hardware"]["device"], "NVIDIA L4")
        self.assertEqual(data["performance_metrics"]["latency_us"], 120.5)

        # Test compact prompt string representation
        prompt_str = pir.to_compact_prompt_str()
        self.assertIn("NVIDIA L4", prompt_str)
        self.assertIn("Long Scoreboard (45.0%)", prompt_str)
        self.assertIn("Latency: 120.50 us", prompt_str)

        # Test round-trip deserialization
        reconstructed = PIRReport.from_dict(data)
        self.assertEqual(reconstructed.kernel_name, pir.kernel_name)
        self.assertEqual(reconstructed.performance_metrics.latency_us, pir.performance_metrics.latency_us)
        self.assertEqual(reconstructed.diagnosis.primary_bottleneck, "MemoryBandwidthBound")


if __name__ == "__main__":
    unittest.main()

