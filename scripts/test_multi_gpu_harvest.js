// Multi-GPU Simulation Test Runner for GPUPerf-3B
const fs = require('fs');
const path = require('path');

const rawDir = path.join(__dirname, '..', 'data', 'raw');
if (!fs.existsSync(rawDir)) fs.mkdirSync(rawDir, { recursive: true });

console.log("=== Simulating Multi-GPU 1-Hour Harvest Runs ===");

const gpus = [
  { device: "Tesla T4", arch: "sm_75", num_sms: 40, bandwidth: 320.0, vram: 16.0 },
  { device: "NVIDIA L4", arch: "sm_89", num_sms: 58, bandwidth: 300.0, vram: 24.0 },
  { device: "NVIDIA A100-SXM4-80GB", arch: "sm_80", num_sms: 108, bandwidth: 2039.0, vram: 80.0 }
];

const operators = ["gemm", "fused_attention", "rmsnorm", "swiglu", "int8_gemm"];

gpus.forEach(gpu => {
  const slug = gpu.device.toLowerCase().replace(/[^a-z0-9]/g, '_');
  const filePath = path.join(rawDir, `harvest_${slug}.jsonl`);
  const records = [];

  console.log(`-> Generating 100 simulated sweeps for ${gpu.device} (${gpu.arch})...`);

  operators.forEach(op => {
    for (let i = 0; i < 20; i++) {
      const tile = [32, 64, 128, 256][i % 4];
      const warps = [2, 4, 8][i % 3];
      
      let speedup = 1.0;
      let bottleneck = "Balanced";
      let spills = 0;

      // GPU-specific physical behaviors
      if (gpu.arch === "sm_75") {
        // T4: Narrow bandwidth and 64KB shared memory
        if (tile >= 128) {
          bottleneck = "MemoryBandwidthBound";
          speedup = 0.78; // Slowdown on T4!
        } else {
          bottleneck = "MemoryBandwidthBound";
          speedup = 1.35; // Downscaled tile wins on T4!
        }
      } else if (gpu.arch === "sm_89") {
        // L4: Register pressure with large unrolling
        if (tile >= 256) {
          bottleneck = "RegisterSpill";
          spills = 2048;
          speedup = 0.65;
        } else {
          speedup = 1.42;
        }
      } else if (gpu.arch === "sm_80") {
        // A100: Huge HBM2e bandwidth, needs large tiles to saturate 108 SMs
        if (tile <= 64) {
          bottleneck = "UnderutilizedSMs";
          speedup = 0.85; // Too small for 108 SMs!
        } else {
          speedup = 1.68; // Large tile wins on A100!
        }
      }

      records.push({
        device: gpu.device,
        arch: gpu.arch,
        num_sms: gpu.num_sms,
        peak_bandwidth_gbs: gpu.bandwidth,
        operator: op,
        config: { BLOCK_M: tile, BLOCK_N: tile, num_warps: warps },
        measured_latency_us: 150.0 / speedup,
        speedup_vs_baseline: speedup,
        pir: {
          kernel_name: `${op}_${gpu.arch}_var${i}`,
          language: "triton",
          hardware: {
            device: gpu.device,
            arch: gpu.arch,
            num_sms: gpu.num_sms,
            peak_dram_bandwidth_gbs: gpu.bandwidth
          },
          launch_config: {
            grid: [Math.ceil(1024 / tile), Math.ceil(1024 / tile), 1],
            block: [warps * 32, 1, 1],
            registers_per_thread: spills > 0 ? 112 : 48,
            spill_loads_bytes: spills,
            spill_stores_bytes: spills
          },
          performance_metrics: {
            latency_us: 150.0 / speedup,
            sol_compute_pct: speedup > 1.0 ? 68.0 : 32.0,
            sol_memory_dram_pct: gpu.bandwidth < 500 ? 88.0 : 42.0,
            achieved_occupancy_pct: spills > 0 ? 22.0 : 58.0,
            theoretical_occupancy_pct: 62.5,
            occupancy_limiter: spills > 0 ? "Registers" : "None"
          },
          diagnosis: {
            primary_bottleneck: bottleneck,
            confidence: 0.92,
            explanation: `Behavior on ${gpu.device}: ${bottleneck}`
          }
        },
        is_chosen: speedup >= 1.15,
        is_rejected: speedup <= 0.90 || spills > 0
      });
    }
  });

  fs.writeFileSync(filePath, records.map(r => JSON.stringify(r)).join('\n') + '\n');
  console.log(`   Written ${records.length} records to ${filePath}`);
});

console.log("\n=== Multi-GPU Data Generation Finished Successfully ===");

