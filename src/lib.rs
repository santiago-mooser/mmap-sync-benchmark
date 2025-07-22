use mmap_sync::synchronizer::Synchronizer;
use rand::Rng;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::ffi::OsString;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;


#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, PartialEq)]
#[archive(check_bytes)]
pub struct BidAsk {
    pub side: String,
    pub exchange: String,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, PartialEq)]
#[archive(check_bytes)]
pub struct BestBidAsk {
    pub best_bid: BidAsk,
    pub best_offer: BidAsk,
}

impl BestBidAsk {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let base_price = rng.gen_range(100.0..200.0);
        let spread = rng.gen_range(0.01..1.0);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let exchanges = ["Binance", "Coinbase", "Kraken", "FTX"];
        let symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"];

        BestBidAsk {
            best_bid: BidAsk {
                side: "buy".to_string(),
                exchange: exchanges[rng.gen_range(0..exchanges.len())].to_string(),
                symbol: symbols[rng.gen_range(0..symbols.len())].to_string(),
                price: base_price - spread / 2.0,
                size: rng.gen_range(0.1..10.0),
                timestamp,
            },
            best_offer: BidAsk {
                side: "sell".to_string(),
                exchange: exchanges[rng.gen_range(0..exchanges.len())].to_string(),
                symbol: symbols[rng.gen_range(0..symbols.len())].to_string(),
                price: base_price + spread / 2.0,
                size: rng.gen_range(0.1..10.0),
                timestamp,
            },
        }
    }

    /// Generate a batch of random BestBidAsk data for benchmarking
    /// This eliminates random generation overhead during actual benchmarks
    pub fn generate_test_data(count: usize) -> Vec<Self> {
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            data.push(Self::new_random());
        }
        data
    }
}

/// Main benchmark setup that manages the mmap synchronizer
pub struct MmapBenchmark {
    path_prefix: Arc<OsString>,
}

impl MmapBenchmark {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let path_prefix = OsString::from("mmap_benchmark");

        // Create an initial synchronizer to set up the mmap file
        let mut sync = Synchronizer::new(&path_prefix);
        let initial_data = BestBidAsk::new_random();
        sync.write(&initial_data, Duration::from_secs(1))?;

        Ok(Self {
            path_prefix: Arc::new(path_prefix),
        })
    }

    /// Creates a new Synchronizer instance for this thread
    pub fn create_synchronizer(&self) -> Result<Synchronizer, Box<dyn std::error::Error>> {
        Ok(Synchronizer::new(&self.path_prefix))
    }

    /// Clean up mmap files - useful between benchmark runs
    pub fn cleanup(&self) -> Result<(), Box<dyn std::error::Error>> {
        // The mmap-sync library should handle cleanup automatically
        // This is a placeholder for any future cleanup needs
        Ok(())
    }
}

impl Clone for MmapBenchmark {
    fn clone(&self) -> Self {
        Self {
            path_prefix: Arc::clone(&self.path_prefix),
        }
    }
}

/// Thread-local worker for mmap operations
/// Separates setup from actual benchmarked operations
pub struct MmapWorker {
    synchronizer: Synchronizer,
}

impl MmapWorker {
    pub fn new(benchmark: &MmapBenchmark) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            synchronizer: benchmark.create_synchronizer()?,
        })
    }

    /// Pure write operation - only measures mmap-sync write time
    /// Does NOT include data generation or serialization setup
    pub fn write_operation(&mut self, data: &BestBidAsk) -> Result<(), Box<dyn std::error::Error>> {
        self.synchronizer.write(data, Duration::from_millis(100))?;
        Ok(())
    }

    /// Pure read operation - only measures mmap-sync read time
    /// Does NOT include deserialization time
    pub fn read_operation(&mut self) -> Result<Option<BestBidAsk>, Box<dyn std::error::Error>> {
        match unsafe { self.synchronizer.read::<BestBidAsk>(false) } {
            Ok(read_result) => {
                let archived = read_result.deref();
                let deserialized: BestBidAsk = archived.deserialize(&mut rkyv::Infallible)?;
                Ok(Some(deserialized))
            }
            Err(_e) => {
                // Convert mmap_sync error to our error type
                Ok(None) // For benchmark purposes, treat read failures as None
            }
        }
    }

    /// Get the underlying synchronizer for advanced operations
    pub fn synchronizer(&mut self) -> &mut Synchronizer {
        &mut self.synchronizer
    }
}

/// Benchmark utilities for setup and measurement
pub mod benchmark_utils {
    use core_affinity::CoreId;
    use rand::Rng;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};
    use crate::{MmapBenchmark, MmapWorker, BestBidAsk};

    const PIN_THREADS_TO_CORE: bool = true;
    const PIN_THREADS_TO_SAME_CORE: bool = true;

    /// Commands for controlling benchmark threads
    #[derive(Debug, Clone)]
    pub enum ThreadCommand {
        StartReading,
        StopReading,
        Shutdown,
        WriteData(BestBidAsk),
    }

    /// Response from benchmark threads
    #[derive(Debug)]
    pub enum ThreadResponse {
        ReadComplete(Duration),
        WriteComplete(Duration),
        ReadyForWork,
        Shutdown,
    }

    /// Thread pool for benchmark operations, eliminates thread creation overhead
    pub struct BenchmarkThreadPool {
        reader_handles: Vec<JoinHandle<()>>,
        reader_senders: Vec<mpsc::Sender<ThreadCommand>>,
        reader_receivers: Vec<mpsc::Receiver<ThreadResponse>>,
        writer_handle: Option<JoinHandle<()>>,
        writer_sender: Option<mpsc::Sender<ThreadCommand>>,
        writer_receiver: Option<mpsc::Receiver<ThreadResponse>>,
        setup_barrier: Arc<Barrier>,
    }

    impl BenchmarkThreadPool {
        /// Create a new thread pool with specified number of readers
        pub fn new(
            benchmark: &MmapBenchmark,
            num_readers: usize,
            reader_frequencies: &[f64]
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let available_cores = Arc::new(get_available_cores());
            let setup_barrier = Arc::new(Barrier::new(num_readers + 2)); // +1 for writer, +1 for coordinator

            let mut reader_handles = Vec::new();
            let mut reader_senders = Vec::new();
            let mut reader_receivers = Vec::new();

            // Create reader threads
            for i in 0..num_readers {
                let (cmd_sender, cmd_receiver) = mpsc::channel();
                let (resp_sender, resp_receiver) = mpsc::channel();

                let benchmark_clone = benchmark.clone();
                let barrier_clone = Arc::clone(&setup_barrier);
                let cores_clone = Arc::clone(&available_cores);
                let frequency = reader_frequencies[i % reader_frequencies.len()];

                let handle = thread::spawn(move || {
                    // Set up thread affinity
                    pin_thread_to_core(i + 1, &cores_clone);

                    // Create worker
                    let mut worker = MmapWorker::new(&benchmark_clone)
                        .expect("Failed to create reader worker");
                    let mut pattern = NormalReaderPattern::new(frequency, READER_STD_DEV_RATIO);

                    // Signal ready
                    barrier_clone.wait();
                    resp_sender.send(ThreadResponse::ReadyForWork).unwrap();

                    // Main worker loop
                    loop {
                        match cmd_receiver.recv() {
                            Ok(ThreadCommand::StartReading) => {
                                let start = Instant::now();
                                let _ = worker.read_operation();
                                let duration = start.elapsed();
                                resp_sender.send(ThreadResponse::ReadComplete(duration)).unwrap();

                                // Wait according to pattern
                                let sleep_duration = pattern.next_inter_arrival_time();
                                thread::sleep(sleep_duration);
                            }
                            Ok(ThreadCommand::StopReading) => {
                                // Just acknowledge, don't do anything
                                resp_sender.send(ThreadResponse::ReadyForWork).unwrap();
                            }
                            Ok(ThreadCommand::Shutdown) => {
                                resp_sender.send(ThreadResponse::Shutdown).unwrap();
                                break;
                            }
                            _ => {} // Ignore other commands
                        }
                    }
                });

                reader_handles.push(handle);
                reader_senders.push(cmd_sender);
                reader_receivers.push(resp_receiver);
            }

            // Create writer thread
            let (writer_cmd_sender, writer_cmd_receiver) = mpsc::channel();
            let (writer_resp_sender, writer_resp_receiver) = mpsc::channel();

            let writer_benchmark = benchmark.clone();
            let writer_barrier = Arc::clone(&setup_barrier);
            let writer_cores = Arc::clone(&available_cores);

            let writer_handle = thread::spawn(move || {
                // Set up thread affinity
                pin_thread_to_core(0, &writer_cores);

                // Create worker
                let mut worker = MmapWorker::new(&writer_benchmark)
                    .expect("Failed to create writer worker");

                // Signal ready
                writer_barrier.wait();
                writer_resp_sender.send(ThreadResponse::ReadyForWork).unwrap();

                // Main worker loop
                loop {
                    match writer_cmd_receiver.recv() {
                        Ok(ThreadCommand::WriteData(data)) => {
                            let start = Instant::now();
                            let _ = worker.write_operation(&data);
                            let duration = start.elapsed();
                            writer_resp_sender.send(ThreadResponse::WriteComplete(duration)).unwrap();
                        }
                        Ok(ThreadCommand::Shutdown) => {
                            writer_resp_sender.send(ThreadResponse::Shutdown).unwrap();
                            break;
                        }
                        _ => {} // Ignore other commands
                    }
                }
            });

            Ok(Self {
                reader_handles,
                reader_senders,
                reader_receivers,
                writer_handle: Some(writer_handle),
                writer_sender: Some(writer_cmd_sender),
                writer_receiver: Some(writer_resp_receiver),
                setup_barrier,
            })
        }

        /// Wait for all threads to be ready
        pub fn wait_for_setup(&self) {
            // Coordinator participates in barrier
            self.setup_barrier.wait();

            // Wait for all ready signals
            if let Some(ref receiver) = self.writer_receiver {
                receiver.recv().unwrap(); // Writer ready signal
            }

            for receiver in &self.reader_receivers {
                receiver.recv().unwrap(); // Reader ready signals
            }
        }

        /// Execute a single write operation and measure latency
        pub fn execute_write(&self, data: &BestBidAsk) -> Result<Duration, Box<dyn std::error::Error>> {
            if let Some(ref sender) = self.writer_sender {
                sender.send(ThreadCommand::WriteData(data.clone()))?;

                if let Some(ref receiver) = self.writer_receiver {
                    match receiver.recv()? {
                        ThreadResponse::WriteComplete(duration) => Ok(duration),
                        _ => Err("Unexpected response from writer thread".into()),
                    }
                } else {
                    Err("Writer receiver not available".into())
                }
            } else {
                Err("Writer sender not available".into())
            }
        }

        /// Start background reading on all reader threads
        pub fn start_background_reading(&self) -> Result<(), Box<dyn std::error::Error>> {
            for sender in &self.reader_senders {
                sender.send(ThreadCommand::StartReading)?;
            }
            Ok(())
        }

        /// Stop background reading
        pub fn stop_background_reading(&self) -> Result<(), Box<dyn std::error::Error>> {
            for sender in &self.reader_senders {
                sender.send(ThreadCommand::StopReading)?;
            }

            // Wait for acknowledgments
            for receiver in &self.reader_receivers {
                receiver.recv()?;
            }
            Ok(())
        }

        /// Execute a single read operation and measure latency
        pub fn execute_read(&self, reader_index: usize) -> Result<Duration, Box<dyn std::error::Error>> {
            if reader_index >= self.reader_senders.len() {
                return Err("Reader index out of bounds".into());
            }

            self.reader_senders[reader_index].send(ThreadCommand::StartReading)?;

            match self.reader_receivers[reader_index].recv()? {
                ThreadResponse::ReadComplete(duration) => Ok(duration),
                _ => Err("Unexpected response from reader thread".into()),
            }
        }

        /// Get number of reader threads
        pub fn num_readers(&self) -> usize {
            self.reader_senders.len()
        }

        /// Shutdown all threads gracefully
        pub fn shutdown(mut self) -> Result<(), Box<dyn std::error::Error>> {
            // Send shutdown commands
            if let Some(ref sender) = self.writer_sender {
                sender.send(ThreadCommand::Shutdown)?;
            }

            for sender in &self.reader_senders {
                sender.send(ThreadCommand::Shutdown)?;
            }

            // Wait for shutdown acknowledgments
            if let Some(ref receiver) = self.writer_receiver {
                receiver.recv()?;
            }

            for receiver in &self.reader_receivers {
                receiver.recv()?;
            }

            // Join all threads
            if let Some(handle) = self.writer_handle.take() {
                handle.join().map_err(|_| "Failed to join writer thread")?;
            }

            for handle in self.reader_handles {
                handle.join().map_err(|_| "Failed to join reader thread")?;
            }

            Ok(())
        }
    }

    /// Statistics collector for multiple benchmark runs
    #[derive(Debug, Clone)]
    pub struct BenchmarkStatistics {
        pub measurements: Vec<Duration>,
        pub runs: Vec<Vec<Duration>>,
    }

    impl BenchmarkStatistics {
        pub fn new() -> Self {
            Self {
                measurements: Vec::new(),
                runs: Vec::new(),
            }
        }

        pub fn add_run(&mut self, run_measurements: Vec<Duration>) {
            self.measurements.extend(run_measurements.iter());
            self.runs.push(run_measurements);
        }

        pub fn mean(&self) -> Duration {
            if self.measurements.is_empty() {
                return Duration::ZERO;
            }
            let total_nanos: u128 = self.measurements.iter()
                .map(|d| d.as_nanos())
                .sum();
            Duration::from_nanos((total_nanos / self.measurements.len() as u128) as u64)
        }

        pub fn std_dev(&self) -> Duration {
            if self.measurements.len() < 2 {
                return Duration::ZERO;
            }

            let mean_nanos = self.mean().as_nanos() as f64;
            let variance: f64 = self.measurements.iter()
                .map(|d| {
                    let diff = d.as_nanos() as f64 - mean_nanos;
                    diff * diff
                })
                .sum::<f64>() / (self.measurements.len() - 1) as f64;

            Duration::from_nanos(variance.sqrt() as u64)
        }

        pub fn coefficient_of_variation(&self) -> f64 {
            let mean_nanos = self.mean().as_nanos() as f64;
            if mean_nanos == 0.0 {
                return 0.0;
            }
            self.std_dev().as_nanos() as f64 / mean_nanos
        }

        pub fn between_run_variance(&self) -> f64 {
            if self.runs.len() < 2 {
                return 0.0;
            }

            let run_means: Vec<f64> = self.runs.iter()
                .map(|run| {
                    let total: u128 = run.iter().map(|d| d.as_nanos()).sum();
                    total as f64 / run.len() as f64
                })
                .collect();

            let overall_mean: f64 = run_means.iter().sum::<f64>() / run_means.len() as f64;

            run_means.iter()
                .map(|mean| (mean - overall_mean).powi(2))
                .sum::<f64>() / (run_means.len() - 1) as f64
        }

        pub fn within_run_variance(&self) -> f64 {
            if self.runs.is_empty() {
                return 0.0;
            }

            let within_run_variances: Vec<f64> = self.runs.iter()
                .map(|run| {
                    if run.len() < 2 {
                        return 0.0;
                    }

                    let run_mean: f64 = run.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / run.len() as f64;
                    run.iter()
                        .map(|d| (d.as_nanos() as f64 - run_mean).powi(2))
                        .sum::<f64>() / (run.len() - 1) as f64
                })
                .collect();

            within_run_variances.iter().sum::<f64>() / within_run_variances.len() as f64
        }
    }

    /// Multi-run benchmark manager
    pub struct MultiRunManager {
        pub num_runs: usize,
        pub samples_per_run: usize,
        pub statistics: BenchmarkStatistics,
    }

    impl MultiRunManager {
        pub fn new(num_runs: usize, samples_per_run: usize) -> Self {
            Self {
                num_runs,
                samples_per_run,
                statistics: BenchmarkStatistics::new(),
            }
        }

        pub fn execute_write_benchmark<F>(&mut self, mut benchmark_fn: F) -> Result<(), Box<dyn std::error::Error>>
        where
            F: FnMut() -> Result<Duration, Box<dyn std::error::Error>>,
        {
            for run in 0..self.num_runs {
                println!("Executing run {}/{}", run + 1, self.num_runs);
                let mut run_measurements = Vec::with_capacity(self.samples_per_run);

                for _ in 0..self.samples_per_run {
                    let duration = benchmark_fn()?;
                    run_measurements.push(duration);
                }

                self.statistics.add_run(run_measurements);
            }
            Ok(())
        }
    }

    // Reader frequency configurations (same as before)
    const READER_STD_DEV_RATIO: f64 = 0.15;

    /// Setup performance environment (CPU affinity, thread priority)
    /// Should be called once before benchmarking, not during measurement
    pub fn setup_performance_environment() -> Result<(), Box<dyn std::error::Error>> {
        // Set thread priority
        if let Err(e) = thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max) {
            eprintln!("Warning: Failed to set high thread priority: {}. Consider running with sudo.", e);
        }

        core_affinity::set_for_current(core_affinity::CoreId { id: 7 });
        Ok(())
    }

    /// Get available CPU cores for thread pinning with detailed diagnostics
    pub fn get_available_cores() -> Vec<CoreId> {
        let physical_cores = num_cpus::get_physical();

        // assume we are using hyperthreading. Multiply by 2 and create a vector of CoreId
        let core_ids: Vec<CoreId> = (0..(physical_cores * 2))
            .map(|id| CoreId { id })
            .collect();

        if core_ids.is_empty() {
            eprintln!("Warning: No logical CPU cores detected. Defaulting to single core.");
            return vec![CoreId { id: 0 }];
        };
        if core_ids.len() < 2 {
            eprintln!("Warning: Only one logical core detected. Disabling thread pinning.");
            return vec![CoreId { id: 0 }];
        };

        if core_ids.len() >= 15 {
            // remove the first three cores to avoid pinning to system-critical cores
            return core_ids[3..].to_vec();
        }
        core_ids
    }

    /// Pin current thread to a specific core (handles insufficient cores gracefully)
    pub fn pin_thread_to_core(core_index: usize, available_cores: &[CoreId]) {

        if !PIN_THREADS_TO_CORE {
            return; // Skip pinning if disabled
        }

        if available_cores.is_empty() {
            println!("Warning: Thread pinning skipped - no available cores");
            return;
        }

        // Smart fallback: disable thread pinning for single-core systems
        // Pinning multiple threads to the same core hurts performance
        if available_cores.len() == 1 {
            return;
        }

        // Use modulo safely for multi-core systems
        let actual_core_index = core_index % available_cores.len();
        let core_id = available_cores[actual_core_index];
        if PIN_THREADS_TO_SAME_CORE{
            // Pin to the same core for all threads
            // println!("Pinning thread to core: {:?}", 14);
            core_affinity::set_for_current(core_affinity::CoreId { id: 6 });
            return;
        }

        // println!("Pinning thread to core: {:?}", core_id);
        core_affinity::set_for_current(core_id);
    }

    /// Normal distribution reader pattern for modeling realistic polling behavior
    /// Each reader polls at a base frequency with normal distribution variation
    pub struct NormalReaderPattern {
        /// Base frequency (Hz)
        base_frequency: f64,
        /// Standard deviation as ratio of mean inter-arrival time
        std_dev_ratio: f64,
        /// Random number generator
        rng: rand::rngs::ThreadRng,
        /// Cached normal random value for Box-Muller transform
        cached_normal: Option<f64>,
    }

    impl NormalReaderPattern {
        /// Create a new normal reader pattern
        /// - base_hz: Target polling frequency in Hz
        /// - std_dev_ratio: Standard deviation as ratio of mean (e.g., 0.1 = 10% variation)
        pub fn new(base_hz: f64, std_dev_ratio: f64) -> Self {
            Self {
                base_frequency: base_hz,
                std_dev_ratio,
                rng: rand::thread_rng(),
                cached_normal: None,
            }
        }

        /// Generate next inter-arrival time using normal distribution
        /// Returns Duration to wait before next read operation
        pub fn next_inter_arrival_time(&mut self) -> Duration {
            let mean_interval = 1.0 / self.base_frequency;
            let std_dev = mean_interval * self.std_dev_ratio;

            // Generate normal random value using Box-Muller transform
            let normal_value = self.generate_normal();

            // Calculate inter-arrival time (ensure it's positive)
            let interval = mean_interval + std_dev * normal_value;
            let positive_interval = interval.max(mean_interval * 0.1); // Min 10% of mean

            Duration::from_secs_f64(positive_interval)
        }

        /// Generate normal random variable using Box-Muller transform
        /// Caches one value for efficiency
        fn generate_normal(&mut self) -> f64 {
            if let Some(cached) = self.cached_normal.take() {
                return cached;
            }

            // Box-Muller transform
            let u1: f64 = self.rng.gen_range(1e-10..1.0); // Avoid log(0)
            let u2: f64 = self.rng.gen_range(0.0..1.0);

            let magnitude = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;

            let z0 = magnitude * angle.cos();
            let z1 = magnitude * angle.sin();

            // Cache one value for next call
            self.cached_normal = Some(z1);
            z0
        }

        /// Get current base frequency
        pub fn base_frequency(&self) -> f64 {
            self.base_frequency
        }
    }

    /// Hawkes process for modeling self-exciting point processes
    /// Used to generate realistic financial market write patterns
    pub struct HawkesProcess {
        /// Baseline intensity (events per second)
        mu: f64,
        /// Self-excitement factor (dimensionless)
        alpha: f64,
        /// Decay rate (1/seconds)
        beta: f64,
        /// Current intensity
        current_intensity: f64,
        /// Last event time (for calculating decay)
        last_event_time: Instant,
        /// Random number generator
        rng: rand::rngs::ThreadRng,
    }

    impl HawkesProcess {
        /// Create a new Hawkes process with financial market parameters
        /// - mu: 10000 Hz baseline
        /// - alpha: 0.3 (moderate self-excitement typical in financial markets)
        /// - beta: 5.0 (quick decay, ~200ms half-life for clustering)
        pub fn new_financial_market() -> Self {
            Self {
                mu: 10000.0,           // 10kHz baseline
                alpha: 0.4,            // 80% self-excitement
                beta: 2.5,            // 100ms clustering decay
                current_intensity: 10000.0,
                last_event_time: Instant::now(),
                rng: rand::thread_rng(),
            }
        }

        /// Generate the next inter-arrival time using thinning algorithm
        /// Returns Duration to wait before next event
        pub fn next_inter_arrival_time(&mut self) -> Duration {
            let now = Instant::now();
            let time_since_last = now.duration_since(self.last_event_time).as_secs_f64();

            // Update current intensity with exponential decay
            self.current_intensity = self.mu + (self.current_intensity - self.mu) * (-self.beta * time_since_last).exp();

            // Use thinning algorithm to generate next event time
            loop {
                // Generate candidate inter-arrival time using current intensity as upper bound
                let u1: f64 = self.rng.gen_range(0.0..1.0);
                let candidate_time = -u1.ln() / self.current_intensity;

                // Calculate actual intensity at candidate time
                let decay_at_candidate = (-self.beta * candidate_time).exp();
                let intensity_at_candidate = self.mu + (self.current_intensity - self.mu) * decay_at_candidate;

                // Accept/reject using thinning
                let u2: f64 = self.rng.gen_range(0.0..1.0);
                if u2 * self.current_intensity <= intensity_at_candidate {
                    // Accepted - update state and return
                    self.last_event_time = now + Duration::from_secs_f64(candidate_time);

                    // Add self-excitement for next event
                    self.current_intensity = intensity_at_candidate + self.alpha;

                    return Duration::from_secs_f64(candidate_time);
                }

                // Rejected - update current intensity and try again
                self.current_intensity = intensity_at_candidate;
            }
        }

        /// Get current intensity (events per second)
        pub fn current_intensity(&self) -> f64 {
            self.current_intensity
        }
    }


}
