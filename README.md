# mmap-sync benchmarking for high-frequency trading

This repo is focused on benchmarking the performance of the [Cloudflare's `mmap-sync` library](https://github.com/cloudflare/mmap-sync) in the context of high-frequency trading systems. The original benchmark provided by cloudflare only focuses on passing a single `bool` value (one byte) to multiple readers, which does not accurately portray how this library would be used in a real-world scenario. In this benchmark, I will be using a more realistic scenario where the writer is sending a struct representing a `BestBidAsk` object (192 bytes) to 12 different readers with different polling intervals.

The test included in this benchmark are:

**Control Experiments:**

- `control_thread_coordination` - Baseline thread coordination overhead
- `control_empty_operations` - Pure mmap operations without concurrent load
- `control_write_scaling` - Write latency scaling analysis (1, 3, 6, 9, 12 readers)
- `control_memory_baseline` - Memory copy operation baselines

**Core Performance Tests:**

- `multi_run_statistical_analysis` - Multi-run variance analysis and stability testing
- `mmap_sync_write_to_many_readers` - Writer latency with multiple concurrent readers
- `mmap_sync_read_from_one_writer` - Reader latency with bursty writer pattern

All benchmarks can be found in the [`benches/mmap_bench.rs`](./benches/mmap_bench.rs) file.

---

- [mmap-sync benchmarking for high-frequency trading](#mmap-sync-benchmarking-for-high-frequency-trading)
- [Benchmarking methodology and testbench setup](#benchmarking-methodology-and-testbench-setup)
  - [Modeling the data](#modeling-the-data)
  - [Modeling the frequency of events](#modeling-the-frequency-of-events)
- [Control experiments and baseline measurements](#control-experiments-and-baseline-measurements)
- [Core benchmarks](#core-benchmarks)
  - [Testbench setup](#testbench-setup)
- [Running the benchmark](#running-the-benchmark)
- [Analyzing the results](#analyzing-the-results)
  - [Read Latency](#read-latency)
  - [Write latency](#write-latency)
- [Confidence in results](#confidence-in-results)
  - [Baseline validation](#baseline-validation)
- [Conclusion](#conclusion)

---

# Benchmarking methodology and testbench setup

The library used to benchmark `mmap-sync` is [`criterion.rs`](https://github.com/bheisler/criterion.rs), which is a rust benchmarking library that provides statistical analysis of the benchmark results. It is also the same library used by Cloudflare in [their benchmarks](https://github.com/cloudflare/mmap-sync?tab=readme-ov-file#benchmarks) of `mmap-sync`.

## Modeling the data

In terms of modeling the data, I used a struct representing a `BestBidAsk` object to represent the best bid and ask prices for a given symbol on an exchange. The `BestBidAsk` struct contains two `BidAsk` objects, one for the best bid and one for the best ask, each containing the side (buy/sell), exchange, symbol, price, size, and timestamp. The data is generated beforehand, then passed to the writer and readers in the benchmark.

```rust
struct BidAsk {
    pub side: String,
    pub exchange: String,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub timestamp: f64,
}
struct BestBidAsk {
    best_bid: BidAsk,
    best_offer: BidAsk,
}
```

## Modeling the frequency of events

In terms of modelling the frequency of events, I used two different distributions:

- For the writer, a [Hawkes distribution](https://wikipedia.org/wiki/Hawkes_process) was used to simulate the bursty nature of market data events, where events tend to cluster together in time.
- For the readers, a [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) was used to model how different trading systems might poll data as quickly as they can process it. This tried to emulate a patterns of multiple readers accessing the data at different rates.

# Control experiments and baseline measurements

The control experiments were used to isolate the actual library performance from system overhead and measurement artifacts.

1. **Thread coordination overhead**
   The `control_thread_coordination` benchmark measures the pure overhead of coordinating multiple threads without any actual mmap operations. This establishes a baseline for the minimum latency introduced by the testing framework itself, allowing us to separate framework overhead from actual mmap-sync performance.
1. **Empty mmap operations**
   The `control_empty_operations` benchmark measures mmap read and write operations in isolation, without any concurrent load from other threads. This provides the theoretical best-case performance of mmap-sync operations and helps identify performance degradation caused by concurrent access patterns.
1. **Write scaling analysis**
   The `control_write_scaling` benchmark systematically tests write latency with 1, 3, 6, 9, and 12 concurrent readers. This scaling analysis reveals if/how write performance degrades as the number of readers increases, providing insight into the scalability characteristics of the mmap-sync library.
1. **Memory baseline**
   The `control_memory_baseline` benchmark measures simple memory copy operations to establish a lower bound for data transfer performance. This baseline helps contextualize mmap-sync performance relative to fundamental memory operations, indicating whether observed latencies are reasonable for the 192-byte data structures being transferred.

# Core benchmarks

1. **Multi-run analysis**
   The `multi_run_statistical_analysis` benchmark executes 5 separate benchmark runs with 200 samples each, enabling variance decomposition analysis. This approach distinguishes between:
   - **Within-run variance**: Natural variation within a single benchmark execution
   - **Between-run variance**: Systematic differences between separate benchmark runs
   - **Stability ratio**: The ratio of between-run to within-run variance (lower values indicate more stable measurements)
   A stability ratio above 0.1 indicates potential measurement instability, suggesting that results may be influenced by external factors rather than true performance characteristics.
1. **Thread pool optimization**
   The benchmarks (`write_to_many_readers` and `read_from_one_writer`) use persistent thread pools to eliminate thread creation overhead from latency measurements. This ensures that measured latencies reflect only the actual mmap-sync operations, not the cost of spawning and coordinating threads during each measurement cycle.
   Additionally, enhanced CPU affinity management and thread priority settings minimize interference from other system processes during benchmarking.

## Testbench setup

The benchmark was run on a desktop computer with the following specifications:

- CPU: AMD Ryzen 7 5800X3D 8-Core Processor
- Memory: 32 GiB Synchronous Unbuffered (Unregistered) 3600 MHz
- Motherboard: X570 AORUS PRO WIFI
- Operating System: Manjaro Linux
- Kernel Version: 6.12.37-1-MANJARO (64-bit)

In order to reduce system jitter, the following kernel parameters were set:

| Kernel parameter                                      | Description                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------- |
| systemd.unit=multi-user.target                        | Disable GUI to remove unnecessary background processes                  |
| initcall_blacklist=amd_pstate_init amd_pstate=disable | Disable  AMD P-state driver to ensure constant CPU frequency            |
| isolcpus=2-7                                          | Isolate CPU cores 2-7 from the scheduler to remove scheduler interrupts |
| nohz_full=2-7                                         | Disable timer interrupts on CPU cores 2-7                               |
| rcu_nocbs=2-7                                         | Disable RCU callbacks on CPU cores 2-7                                  |
| idle=poll                                             | Disable CPU idle states                                                 |
| nmi_watchdog=0                                        | Disable NMI watchdog                                                    |
| watchdog=none                                         | Disable watchdog                                                        |
| audit=0                                               | Disable audit subsystem                                                 |

Additionally, the following commands were run to move system threads and workqueues to CPU core 0 in order to prevent them from interfering with the benchmark as well as setting the CPU governor to "performance" mode to ensure the CPU doesn't change between different frequencies:

```bash
pgrep -P 2 | xargs -i taskset -p -c 0 {}
find /sys/devices/virtual/workqueue -name cpumask  -exec sh -c 'echo 1 > {}' ';'
sudo tee "performance" /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
```

# Running the benchmark

The benchmark itself was run with the following command, which enables rust optimizations and set the niceness level to -20 (highest priority):

```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C lto" sudo nice cargo bench
```

# Analyzing the results

Looking at the results, both the read and write latencies are higher than in the original benchmark. However, this is expected since we are sending a much larger amount of data (192 bytes vs 1 byte). Initially the both read and write latencies appear to be quite consistent, with the mean and median being relatively consistent. However, looking more closely at the write latencies, we can see that the standard deviation is quite high, which indicates potential latency issues (more on this later).

| Read latency |                |                |                |     | Write latency |               |               |               |
| ------------ | -------------- | -------------- | -------------- | --- | ------------- | ------------- | ------------- | ------------- |
|              | Lower bound    | Estimate       | Upper bound    |     |               | Lower bound   | Estimate      | Upper bound   |
| Throughput   | 32.271 Kelem/s | 32.293 Kelem/s | 32.312 Kelem/s |     | Throughput    | 153.09 elem/s | 153.86 elem/s | 154.63 elem/s |
| R²           | 0.0000532      | 0.0000533      | 0.0000532      |     | R²            | 0.0000304     | 0.0000305     | 0.0000304     |
| Mean         | 30.948 µs      | 30.966 µs      | 30.987 µs      |     | Mean          | 6.4669 ms     | 6.4992 ms     | 6.5321 ms     |
| Std. Dev.    | 523.50 ns      | 703.88 ns      | 956.43 ns      |     | Std. Dev.     | 1.0984 ms     | 1.1769 ms     | 1.2743 ms     |
| Median       | 30.809 µs      | 30.836 µs      | 30.861 µs      |     | Median        | 6.0735 ms     | 6.0762 ms     | 6.0796 ms     |
| MAD          | 477.58 ns      | 500.38 ns      | 518.54 ns      |     | MAD           | 81.561 µs     | 85.246 µs     | 88.877 µs     |

## Read Latency

Looking at the latency graph of the readers (where the write pattern follows a Hawkes distribution), we can see that the read latency remains mostly remains relatively consistent, with only one or two spikes in the latency. The undulating nature of the benchmark is related to the Hawkes distribution, which causes the writer to send bursts of data at random intervals. The readers are able to keep up with the writer, and the read latency remains relatively low (around 30-32 µs). The standard deviation of the read latency is also relatively low, at around 523 ns, which indicates that the read latency is fairly consistent. There are only 132 outliers (2.64%) in the read latency (spikes above 2 standard deviations), which is a good sign that the readers are able to keep up with the writer.

```ascii
mmap_sync_read_from_one_writer/read_with_concurrent_writer
                        time:   [30.948 µs 30.966 µs 30.987 µs]
                        thrpt:  [32.271 Kelem/s 32.293 Kelem/s 32.312 Kelem/s]
                 change:
                        time:   [+12.323% +12.764% +13.163%] (p = 0.00 < 0.05)
                        thrpt:  [-11.632% -11.320% -10.971%]
                        Performance has regressed.
Found 132 outliers among 5000 measurements (2.64%)
  116 (2.32%) high mild
  16 (0.32%) high severe
```

<p float="left">
  <img alt="Write Latency" src="./report/mmap_sync_read_from_one_writer/read_with_concurrent_writer/report/iteration_times.svg"  width="960" />
  <img alt="Write Latency PDF" src="./report/mmap_sync_read_from_one_writer/read_with_concurrent_writer/report/pdf.svg"  width="960" />
</p>

## Write latency

Things take a turn for the worse when we look at the write latency. It's expected that the write latency is significantly higher than the read latency, but what is not expected is the huge spikes in latency encountered at semi-regular intervals. While the base latency for writing stays around 6.2ms, there are large and consistent spikes above 7ms, reaching up to 14ms. These account for 57 (1.14%) spikes between two and three standard deviations and a massive 746 (14.92%) latency spikes above three standard deviations. This a 233% increase in latency, which is unacceptable for a high-frequency trading system.

```ascii
mmap_sync_write_to_many_readers/write_with_concurrent_readers/12
                        time:   [6.4669 ms 6.4992 ms 6.5321 ms]
                        thrpt:  [153.09  elem/s 153.86  elem/s 154.63  elem/s]
                 change:
                        time:   [-5.2303% -4.4688% -3.6923%] (p = 0.00 < 0.05)
                        thrpt:  [+3.8338% +4.6779% +5.5189%]
                        Performance has improved.
Found 804 outliers among 5000 measurements (16.08%)
  1 (0.02%) low mild
  57 (1.14%) high mild
  746 (14.92%) high severe
```

<p float="left">
  <img alt="Write Latency Graph" src="./report/mmap_sync_write_to_many_readers/write_with_concurrent_readers/12/report/iteration_times.svg"  width="960" />
  <img alt="Write Latency PDF" src="./report/mmap_sync_write_to_many_readers/write_with_concurrent_readers/12/report/pdf.svg"  width="960" />
</p>

Scaling the number of readers from 1 to 12 does not seem to increase write latency, but we can see that there is still a large amount of variance in the write latency regardless of the number of readers:

<p float="left">
  <img alt="Write Latency PDF with reader scaling" src="./report/control_write_scaling/report/violin.svg"  width="960" />
</p>

# Confidence in results

## Baseline validation

The control experiments establish that the write latency spikes are genuine characteristics of mmap-sync performance rather than measurement artifacts:

- **Thread coordination overhead** measurements confirm that the testing framework introduces minimal latency (typically <1µs), meaning the 6.5ms write latencies are attributable to mmap-sync operations
- **Empty mmap operations** provide baseline performance without concurrent load, allowing isolation of concurrency-related performance degradation
- **Memory baseline** operations demonstrate that simple memory copies are orders of magnitude faster than observed mmap-sync latencies, confirming that the library introduces substantial overhead

# Conclusion

The comprehensive benchmarking analysis, provides high confidence in the performance assessment of the `mmap-sync` library for high-frequency trading applications. This aligns with industry knowledge, where other types of Inter-process Communication (IPC) mechanisms like shared memory (e.g. ring queues) are typically preferred for their lower latencies.

If you are interested in the report (which contains additional statistics about the benchmark), you can find it in the `report` directory. Additionally, the raw benchmark data can be found in the `data` directory.
