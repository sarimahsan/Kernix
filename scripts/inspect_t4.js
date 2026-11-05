const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, '..', 'harvest_t4_sm_75.jsonl');
const lines = fs.readFileSync(filePath, 'utf-8').split('\n').filter(Boolean);

console.log('Total entries in file:', lines.length);

const mockRecords = lines.slice(0, 504).map(l => JSON.parse(l));
const realRecords = lines.slice(504).map(l => JSON.parse(l));

console.log('Mock records (run 1):', mockRecords.length);
console.log('Real physical GPU records (run 2):', realRecords.length);

const realChosen = realRecords.filter(r => r.is_chosen);
const realRejected = realRecords.filter(r => r.is_rejected);

console.log('Real winners (chosen, speedup >= 1.15x):', realChosen.length);
console.log('Real pitfalls (rejected, speedup <= 0.88x):', realRejected.length);

realChosen.sort((a, b) => b.speedup_vs_baseline - a.speedup_vs_baseline);

console.log('\n--- Top 3 Fastest Physical Configurations on Tesla T4 ---');
realChosen.slice(0, 3).forEach((r, i) => {
  console.log(`\nRank #${i + 1}:`);
  console.log('  Operator:      ', r.operator);
  console.log('  Configuration: ', JSON.stringify(r.config));
  console.log('  Measured Time: ', `${r.measured_latency_us.toFixed(1)} us (~${(r.measured_latency_us / 1000).toFixed(2)} ms)`);
  console.log('  Speedup:       ', `${r.speedup_vs_baseline}x`);
  console.log('  Compute SOL:   ', `${r.pir.performance_metrics.sol_compute_pct}%`);
  console.log('  DRAM SOL:      ', `${r.pir.performance_metrics.sol_memory_dram_pct}%`);
});

realRejected.sort((a, b) => a.speedup_vs_baseline - b.speedup_vs_baseline);

console.log('\n--- Worst 3 Slowdown Pitfalls on Tesla T4 ---');
realRejected.slice(0, 3).forEach((r, i) => {
  console.log(`\nPitfall #${i + 1}:`);
  console.log('  Operator:      ', r.operator);
  console.log('  Configuration: ', JSON.stringify(r.config));
  console.log('  Measured Time: ', `${r.measured_latency_us.toFixed(1)} us (~${(r.measured_latency_us / 1000).toFixed(2)} ms)`);
  console.log('  Slowdown:      ', `${r.speedup_vs_baseline}x`);
  console.log('  Diagnosis:     ', r.pir.diagnosis.explanation);
});

// Clean up the file to keep ONLY the real physical records
const cleanedPath = path.join(__dirname, '..', 'data', 'raw', 'harvest_tesla_t4_real.jsonl');
fs.writeFileSync(cleanedPath, realRecords.map(r => JSON.stringify(r)).join('\n') + '\n');
console.log(`\n[+] Saved cleaned 100% real GPU dataset to ${cleanedPath} (${realRecords.length} records)`);

