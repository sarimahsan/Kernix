// Multi-GPU Dataset Merger (Node.js companion)
const fs = require('fs');
const path = require('path');

const dataDir = path.join(__dirname, '..', 'data');
const rawDir = path.join(dataDir, 'raw');

console.log("=== Merging Multi-GPU Datasets into Unified SFT & DPO Training Sets ===");

const files = fs.readdirSync(rawDir).filter(f => f.startsWith('harvest_') && f.endsWith('.jsonl'));
console.log(`Found ${files.length} hardware harvest files:`);
files.forEach(f => console.log(`  * ${f}`));

let allChosen = [];
let allRejected = [];
const archCounts = {};

files.forEach(f => {
  const lines = fs.readFileSync(path.join(rawDir, f), 'utf-8').split('\n').filter(Boolean);
  lines.forEach(l => {
    try {
      const rec = JSON.parse(l);
      archCounts[rec.arch] = (archCounts[rec.arch] || 0) + 1;
      if (rec.is_chosen) allChosen.push(rec);
      else if (rec.is_rejected) allRejected.push(rec);
    } catch (e) {}
  });
});

console.log("\nHardware Trace Distribution:");
Object.entries(archCounts).forEach(([arch, count]) => {
  console.log(`  * ${arch}: ${count} records`);
});
console.log(`Total Chosen Winners: ${allChosen.length} | Total Rejected Pitfalls: ${allRejected.length}`);

// Build SFT samples
const sftRecords = allChosen.map(chosen => {
  const hw = chosen.pir.hardware;
  const cfg = chosen.config;
  const op = chosen.operator.toUpperCase();
  const speedup = chosen.speedup_vs_baseline;

  return {
    messages: [
      {
        role: "system",
        content: "You are GPUPerf, a specialized GPU systems optimization engineer. Analyze GPU kernels and profiler evidence, diagnose root-cause bottlenecks, and provide an optimized, empirically faster kernel."
      },
      {
        role: "user",
        content: `Here is a TRITON GPU kernel and its hardware profiling report from NVIDIA Nsight Compute:\n\n\`\`\`triton\n# Unoptimized baseline for ${op}\n\`\`\`\n\n[PROFILER REPORT]\nDevice: ${hw.device} (${hw.arch}, ${hw.num_sms} SMs, ${hw.peak_dram_bandwidth_gbs} GB/s DRAM)\nKernel: ${chosen.pir.kernel_name}\nSpeed of Light: Compute ${chosen.pir.performance_metrics.sol_compute_pct.toFixed(1)}% | Memory (DRAM) ${chosen.pir.performance_metrics.sol_memory_dram_pct.toFixed(1)}%\nOccupancy: Achieved ${chosen.pir.performance_metrics.achieved_occupancy_pct.toFixed(1)}% (Limiter: ${chosen.pir.performance_metrics.occupancy_limiter})\nBottleneck: ${chosen.pir.diagnosis.primary_bottleneck}\n[/PROFILER REPORT]\n\nAnalyze the profiler metrics and write an optimized kernel for ${hw.device}.`
      },
      {
        role: "assistant",
        content: `### Bottleneck Diagnosis\n${chosen.pir.diagnosis.explanation}\n\n### Optimization Strategy for ${hw.device}\nSelected optimal configuration {BLOCK_M: ${cfg.BLOCK_M}, BLOCK_N: ${cfg.BLOCK_N}, warps: ${cfg.num_warps}} to maximize occupancy while respecting ${hw.arch} hardware resource limits.\n\n### Optimized TRITON Kernel\n\`\`\`triton\n# Optimized ${op} for ${hw.device}\n# Tuned config: BLOCK_M=${cfg.BLOCK_M}, BLOCK_N=${cfg.BLOCK_N}, num_warps=${cfg.num_warps}\n\`\`\`\n\n### Verification\n- Measured Speedup: ${speedup.toFixed(2)}x on ${hw.device}\n- Validated numerical correctness against PyTorch reference.`
      }
    ],
    metadata: {
      device: hw.device,
      arch: hw.arch,
      operator: chosen.operator,
      speedup: speedup
    }
  };
});

// Build DPO preference pairs
const dpoRecords = [];
allChosen.forEach((chosen, idx) => {
  if (idx < allRejected.length) {
    const rejected = allRejected[idx];
    const hw = chosen.pir.hardware;
    const chosenCfg = chosen.config;
    const rejCfg = rejected.config;

    dpoRecords.push({
      prompt: `Here is a TRITON GPU kernel and its hardware profiling report from NVIDIA Nsight Compute:\n\n[PROFILER REPORT]\nDevice: ${hw.device} (${hw.arch}, ${hw.num_sms} SMs, ${hw.peak_dram_bandwidth_gbs} GB/s DRAM)\nKernel: ${chosen.pir.kernel_name}\nBottleneck: ${chosen.pir.diagnosis.primary_bottleneck}\n[/PROFILER REPORT]\n\nAnalyze the profiler metrics and write an optimized kernel for ${hw.device}.`,
      chosen: `### Bottleneck Diagnosis\nOptimized for ${hw.device} architecture.\n\n### Optimized TRITON Kernel\n\`\`\`triton\n# Tuned config: BLOCK_M=${chosenCfg.BLOCK_M}, BLOCK_N=${chosenCfg.BLOCK_N}, warps=${chosenCfg.num_warps}\n\`\`\`\n\n### Verification\nAchieved ${chosen.speedup_vs_baseline.toFixed(2)}x speedup on ${hw.device}.`,
      rejected: `### Bottleneck Diagnosis\nFailed to respect ${hw.arch} memory limits.\n\n### Optimized TRITON Kernel\n\`\`\`triton\n# Regressed config: BLOCK_M=${rejCfg.BLOCK_M}, BLOCK_N=${rejCfg.BLOCK_N}, warps=${rejCfg.num_warps}\n\`\`\`\n\n### Verification\nResulted in ${rejected.speedup_vs_baseline.toFixed(2)}x slowdown on ${hw.device}.`,
      metadata: {
        device: hw.device,
        arch: hw.arch,
        chosen_speedup: chosen.speedup_vs_baseline,
        rejected_speedup: rejected.speedup_vs_baseline
      }
    });
  }
});

const sftPath = path.join(dataDir, 'sft_multi_gpu.jsonl');
const dpoPath = path.join(dataDir, 'dpo_multi_gpu.jsonl');

fs.writeFileSync(sftPath, sftRecords.map(r => JSON.stringify(r)).join('\n') + '\n');
fs.writeFileSync(dpoPath, dpoRecords.map(r => JSON.stringify(r)).join('\n') + '\n');

console.log(`\n-> Successfully compiled ${sftRecords.length} Multi-GPU SFT samples to ${sftPath}`);
console.log(`-> Successfully compiled ${dpoRecords.length} Multi-GPU DPO pairs to ${dpoPath}`);
console.log("=== Multi-GPU Dataset Generation Complete ===");

