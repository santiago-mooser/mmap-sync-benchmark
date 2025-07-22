use benchmark_utils::{BenchmarkThreadPool, MultiRunManager};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mmap_benchmark::{BestBidAsk, MmapBenchmark, MmapWorker, benchmark_utils};
use std::time::Duration;

const NUM_READERS: usize = 12;
const SAMPLE_SIZE: usize = 5000; // Reduced for multiple runs
const NUM_RUNS: usize = 5; // Multiple runs for statistical analysis

// Reader frequency configurations (750Hz to 10kHz range)
const READER_FREQUENCIES: &[f64] = &[750.0, 1500.0, 2500.0, 5000.0, 10000.0]; // Hz

/// Setup function called once before all benchmarks
fn setup_benchmarks() {
    if let Err(e) = benchmark_utils::setup_performance_environment() {
        eprintln!("Warning: Failed to setup performance environment: {}", e);
    }
}

/// Control experiment: Measure thread coordination overhead without mmap operations
fn benchmark_thread_coordination_overhead(c: &mut Criterion) {
    setup_benchmarks();

    let mut group = c.benchmark_group("control_thread_coordination");
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(5));

    group.bench_function("coordination_overhead", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;

            for _ in 0..iters {
                let start = std::time::Instant::now();
                // Just measure the time to do thread coordination without actual work
                std::thread::sleep(Duration::from_nanos(1)); // Minimal work
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.finish();
}

/// Control experiment: Measure empty mmap operations (no concurrent load)
fn benchmark_empty_operations(c: &mut Criterion) {
    setup_benchmarks();

    let mut group = c.benchmark_group("control_empty_operations");
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(5));

    let benchmark = MmapBenchmark::new().expect("Failed to create benchmark");
    let test_data = BestBidAsk::generate_test_data(100);

    group.bench_function("empty_write", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;
            let mut worker = MmapWorker::new(&benchmark).expect("Failed to create worker");

            for i in 0..iters {
                let data = &test_data[i as usize % test_data.len()];
                let start = std::time::Instant::now();
                worker.write_operation(data).expect("Write failed");
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.bench_function("empty_read", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;
            let mut worker = MmapWorker::new(&benchmark).expect("Failed to create worker");

            for _ in 0..iters {
                let start = std::time::Instant::now();
                let _ = worker.read_operation().expect("Read failed");
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.finish();
}

/// Control experiment: Test write latency scaling with different numbers of readers
fn benchmark_write_scaling(c: &mut Criterion) {
    setup_benchmarks();

    let mut group = c.benchmark_group("control_write_scaling");
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(10));

    let benchmark = MmapBenchmark::new().expect("Failed to create benchmark");
    let test_data = BestBidAsk::generate_test_data(100);

    // Test with different numbers of readers: 1, 3, 6, 9, 12
    for num_readers in [1, 3, 6, 9, 12] {
        group.bench_function(BenchmarkId::new("write_latency", num_readers), |b| {
            b.iter_custom(|iters| {
                // Create thread pool once outside measurement
                let thread_pool =
                    BenchmarkThreadPool::new(&benchmark, num_readers, READER_FREQUENCIES)
                        .expect("Failed to create thread pool");

                thread_pool.wait_for_setup();

                // Start background reading
                thread_pool
                    .start_background_reading()
                    .expect("Failed to start reading");

                let mut total_time = Duration::ZERO;

                // Measure only the write operations
                for i in 0..iters {
                    let data = &test_data[i as usize % test_data.len()];
                    let write_time = thread_pool.execute_write(data).expect("Write failed");
                    total_time += write_time;
                }

                // Clean shutdown
                thread_pool
                    .stop_background_reading()
                    .expect("Failed to stop reading");
                thread_pool
                    .shutdown()
                    .expect("Failed to shutdown thread pool");

                total_time
            });
        });
    }

    group.finish();
}

///  write benchmark with thread pool reuse and multiple runs
fn benchmark_write_to_many_readers(c: &mut Criterion) {
    setup_benchmarks();

    let mut group = c.benchmark_group("write_to_many_readers");
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(10));

    let benchmark = MmapBenchmark::new().expect("Failed to create benchmark");
    let test_data = BestBidAsk::generate_test_data(1000);

    group.bench_function(BenchmarkId::new("multi_run_write", NUM_READERS), |b| {
        b.iter_custom(|iters| {
            // Create thread pool once, reuse for all measurements
            let thread_pool = BenchmarkThreadPool::new(&benchmark, NUM_READERS, READER_FREQUENCIES)
                .expect("Failed to create thread pool");

            thread_pool.wait_for_setup();
            thread_pool
                .start_background_reading()
                .expect("Failed to start reading");

            let mut total_time = Duration::ZERO;

            // Multiple measurement runs for better statistics
            for i in 0..iters {
                let data = &test_data[i as usize % test_data.len()];
                let write_time = thread_pool.execute_write(data).expect("Write failed");
                total_time += write_time;
            }

            thread_pool
                .stop_background_reading()
                .expect("Failed to stop reading");
            thread_pool
                .shutdown()
                .expect("Failed to shutdown thread pool");

            total_time
        });
    });

    group.finish();
}

///  read benchmark with thread pool reuse
fn benchmark_read_from_one_writer(c: &mut Criterion) {
    setup_benchmarks();

    let mut group = c.benchmark_group("read_from_one_writer");
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(10));

    let benchmark = MmapBenchmark::new().expect("Failed to create benchmark");
    let test_data = BestBidAsk::generate_test_data(1000);

    group.bench_function("multi_run_read", |b| {
        b.iter_custom(|iters| {
            // Create thread pool with writer for background writes
            let thread_pool = BenchmarkThreadPool::new(
                &benchmark,
                1,         // Single reader for this test
                &[1000.0], // 1kHz reader frequency
            )
            .expect("Failed to create thread pool");

            thread_pool.wait_for_setup();

            let mut total_time = Duration::ZERO;

            // Start background writing pattern (Hawkes process simulation)
            std::thread::spawn({
                let benchmark_clone = benchmark.clone();
                let test_data_clone = test_data.clone();
                move || {
                    let mut writer =
                        MmapWorker::new(&benchmark_clone).expect("Failed to create writer");
                    let mut hawkes = benchmark_utils::HawkesProcess::new_financial_market();
                    let mut data_index = 0;

                    for _ in 0..1000 {
                        // Background writes during the test
                        let data = &test_data_clone[data_index % test_data_clone.len()];
                        let _ = writer.write_operation(data);
                        data_index += 1;

                        let sleep_duration = hawkes.next_inter_arrival_time();
                        std::thread::sleep(sleep_duration);
                    }
                }
            });

            // Small delay to let background writer start
            std::thread::sleep(Duration::from_millis(10));

            // Measure read operations
            for _ in 0..iters {
                let read_time = thread_pool.execute_read(0).expect("Read failed");
                total_time += read_time;
            }

            thread_pool
                .shutdown()
                .expect("Failed to shutdown thread pool");

            total_time
        });
    });

    group.finish();
}

/// Benchmark memory baseline operations for comparison
fn benchmark_memory_baseline(c: &mut Criterion) {
    setup_benchmarks();

    let mut group = c.benchmark_group("control_memory_baseline");
    group.throughput(Throughput::Elements(1));
    group.sample_size(SAMPLE_SIZE);

    // Generate test data
    let source_data = BestBidAsk::generate_test_data(100);

    group.bench_function("memory_copy", |b| {
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;
            let mut destination = Vec::with_capacity(iters as usize);

            for i in 0..iters {
                let data = &source_data[i as usize % source_data.len()];
                let start = std::time::Instant::now();
                destination.push(data.clone());
                total_time += start.elapsed();
            }

            total_time
        });
    });

    group.finish();
}

criterion_group!(
    control_experiments,
    benchmark_thread_coordination_overhead,
    benchmark_empty_operations,
    benchmark_write_scaling,
    benchmark_memory_baseline,
);

criterion_group!(
    benchmarks,
    benchmark_write_to_many_readers,
    benchmark_read_from_one_writer,
);

criterion_main!(control_experiments, benchmarks);
