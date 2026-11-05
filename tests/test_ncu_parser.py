"""Unit tests for the NCU Parser and Bottleneck Diagnoser."""

import unittest
from gpuperf.profiler.ncu_parser import NCUParser
from gpuperf.schema.pir import PIRReport, HardwareInfo, LaunchConfig, PerformanceMetrics


class TestNCUParser(unittest.TestCase):
    def test_csv_parsing_and_register_spill_diagnosis(self):
        csv_sample = """Metric Name,Metric Value
gpu__time_duration.sum,250000
sm__throughput.avg.pct_of_peak_sustained_elapsed,20.0
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,30.0
sm__warps_active.avg.pct_of_peak_sustained_active,15.0
sm__maximum_warps_per_active_cycle_pct,25.0
launch__registers_per_thread,120
launch__shared_mem_per_block_allocated,16384
smsp__sass_inst_executed_op_local_ld.sum,2048
smsp__sass_inst_executed_op_local_st.sum,2048
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,80.0
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,80.0
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,0
"""
        pir = NCUParser.parse_csv(csv_sample, kernel_name="spilling_kernel", language="triton")

        self.assertEqual(pir.kernel_name, "spilling_kernel")
        self.assertEqual(pir.launch_config.spill_loads_bytes, 2048)
        self.assertEqual(pir.launch_config.spill_stores_bytes, 2048)
        self.assertEqual(pir.performance_metrics.latency_us, 250.0)

        # Bottleneck diagnosis should flag register spill
        self.assertIsNotNone(pir.diagnosis)
        self.assertEqual(pir.diagnosis.primary_bottleneck, "RegisterSpill")
        self.assertGreater(pir.diagnosis.confidence, 0.95)

    def test_memory_bandwidth_bound_diagnosis(self):
        csv_sample = """Metric Name,Metric Value
gpu__time_duration.sum,150000
sm__throughput.avg.pct_of_peak_sustained_elapsed,25.0
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,85.0
sm__warps_active.avg.pct_of_peak_sustained_active,40.0
sm__maximum_warps_per_active_cycle_pct,50.0
launch__registers_per_thread,32
launch__shared_mem_per_block_allocated,4096
smsp__sass_inst_executed_op_local_ld.sum,0
smsp__sass_inst_executed_op_local_st.sum,0
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,60.0
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,60.0
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,0
"""
        pir = NCUParser.parse_csv(csv_sample, kernel_name="bandwidth_kernel", language="triton")
        self.assertEqual(pir.diagnosis.primary_bottleneck, "MemoryBandwidthBound")


if __name__ == "__main__":
    unittest.main()

