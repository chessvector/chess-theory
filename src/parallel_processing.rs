/*
Parallel Processing and Progress Visualization Module

This module provides parallel execution capabilities and progress visualization
for long-running mathematical discovery processes.

Mathematical Framework:
- Parallel Pattern Discovery: Π_parallel: ℝ^{n×m} → 𝒫(Patterns) with thread-level decomposition
- Batch Processing: E*_batch({s₁, s₂, ..., sₙ}) with concurrent evaluation
- Progress Tracking: Real-time visualization of discovery convergence
*/

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use anyhow::Result;
use tokio::time::sleep;
use futures::future::join_all;
use crossbeam::channel::{self, Receiver, Sender};

use crate::ChessPosition;
use crate::stockfish_oracle::{StockfishOracle, StockfishEvaluation};
use crate::discovery_engine::{ChessMathematicalDiscoveryEngine, DiscoveryResults};
use ndarray::Array1;

/// Parallel Processing Configuration
/// Mathematical form: Configuration for thread-level decomposition
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads for pattern discovery
    pub discovery_threads: usize,
    
    /// Number of concurrent oracle evaluations
    pub oracle_concurrency: usize,
    
    /// Batch size for position processing
    pub position_batch_size: usize,
    
    /// Enable progress bars
    pub show_progress: bool,
    
    /// Progress update frequency in milliseconds
    pub progress_update_ms: u64,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            discovery_threads: num_cpus::get(),
            oracle_concurrency: 4, // Reasonable for Stockfish
            position_batch_size: 50,
            show_progress: true,
            progress_update_ms: 100,
        }
    }
}

/// Progress Tracker for Mathematical Discovery
/// Tracks convergence progress across multiple dimensions
pub struct DiscoveryProgressTracker {
    /// Multi-progress container for multiple progress bars
    multi_progress: MultiProgress,
    
    /// Main discovery progress bar
    main_progress: ProgressBar,
    
    /// Oracle evaluation progress bar
    oracle_progress: ProgressBar,
    
    /// Pattern discovery progress bar
    pattern_progress: ProgressBar,
    
    /// Knowledge base consistency progress bar
    kb_progress: ProgressBar,
    
    /// Total operations counter
    total_operations: Arc<Mutex<usize>>,
    
    /// Completed operations counter
    completed_operations: Arc<Mutex<usize>>,
}

impl DiscoveryProgressTracker {
    /// Creates a new progress tracker for discovery processes
    pub fn new(total_positions: usize, total_iterations: usize) -> Self {
        let multi_progress = MultiProgress::new();
        
        // Main discovery progress
        let main_progress = multi_progress.add(ProgressBar::new(total_iterations as u64));
        main_progress.set_style(
            ProgressStyle::default_bar()
                .template("🔬 Discovery Progress: [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations ({percent}%) ETA: {eta}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        main_progress.set_message("Mathematical Discovery Engine");
        
        // Oracle evaluation progress
        let oracle_progress = multi_progress.add(ProgressBar::new((total_positions * total_iterations) as u64));
        oracle_progress.set_style(
            ProgressStyle::default_bar()
                .template("🎯 Oracle Evaluations: [{elapsed_precise}] [{bar:40.green/yellow}] {pos}/{len} positions ({percent}%) Rate: {per_sec}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        oracle_progress.set_message("Stockfish E*(s) Evaluations");
        
        // Pattern discovery progress
        let pattern_progress = multi_progress.add(ProgressBar::new(total_positions as u64));
        pattern_progress.set_style(
            ProgressStyle::default_bar()
                .template("🔍 Pattern Discovery: [{elapsed_precise}] [{bar:40.magenta/red}] {pos}/{len} patterns ({percent}%) Found: {msg}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        pattern_progress.set_message("0 strategic patterns");
        
        // Knowledge base consistency progress
        let kb_progress = multi_progress.add(ProgressBar::new(100));
        kb_progress.set_style(
            ProgressStyle::default_bar()
                .template("📊 KB Consistency: [{elapsed_precise}] [{bar:40.yellow/blue}] {percent}% consistency Score: {msg}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        kb_progress.set_message("0.000");
        
        Self {
            multi_progress,
            main_progress,
            oracle_progress,
            pattern_progress,
            kb_progress,
            total_operations: Arc::new(Mutex::new(0)),
            completed_operations: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Updates discovery iteration progress
    pub fn update_iteration(&self, iteration: usize, convergence_score: f64) {
        self.main_progress.set_position(iteration as u64);
        self.main_progress.set_message(format!("Convergence: {:.2}%", convergence_score * 100.0));
    }
    
    /// Updates oracle evaluation progress
    pub fn update_oracle(&self, completed: usize, total: usize) {
        self.oracle_progress.set_position(completed as u64);
        if total > 0 {
            self.oracle_progress.set_length(total as u64);
        }
    }
    
    /// Updates pattern discovery progress
    pub fn update_patterns(&self, processed: usize, patterns_found: usize) {
        self.pattern_progress.set_position(processed as u64);
        self.pattern_progress.set_message(format!("{} strategic patterns", patterns_found));
    }
    
    /// Updates knowledge base consistency
    pub fn update_kb_consistency(&self, consistency_score: f64) {
        let position = (consistency_score * 100.0) as u64;
        self.kb_progress.set_position(position);
        self.kb_progress.set_message(format!("{:.3}", consistency_score));
    }
    
    /// Finishes all progress bars
    pub fn finish(&self) {
        self.main_progress.finish_with_message("Discovery Complete");
        self.oracle_progress.finish_with_message("Oracle Evaluations Complete");
        self.pattern_progress.finish_with_message("Pattern Discovery Complete");
        self.kb_progress.finish_with_message("Consistency Analysis Complete");
    }
    
    /// Gets the multi-progress handle for custom progress bars
    pub fn multi_progress(&self) -> &MultiProgress {
        &self.multi_progress
    }
}

/// Parallel Oracle Evaluator
/// Provides concurrent Stockfish evaluation with progress tracking
pub struct ParallelOracleEvaluator {
    config: ParallelConfig,
    progress_tracker: Option<Arc<DiscoveryProgressTracker>>,
}

impl ParallelOracleEvaluator {
    /// Creates a new parallel oracle evaluator
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            progress_tracker: None,
        }
    }
    
    /// Sets progress tracker for evaluation monitoring
    pub fn with_progress_tracker(mut self, tracker: Arc<DiscoveryProgressTracker>) -> Self {
        self.progress_tracker = Some(tracker);
        self
    }
    
    /// Evaluates positions in parallel with progress tracking
    /// Mathematical form: E*_parallel({s₁, s₂, ..., sₙ}) → {e₁, e₂, ..., eₙ}
    pub async fn evaluate_batch_parallel(
        &self,
        oracle: &mut StockfishOracle,
        positions: &[ChessPosition],
    ) -> Result<Vec<StockfishEvaluation>> {
        
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        
        let total_positions = positions.len();
        
        // Update progress tracker
        if let Some(tracker) = &self.progress_tracker {
            tracker.update_oracle(0, total_positions);
        }
        
        // Split positions into batches for parallel processing
        let batches: Vec<&[ChessPosition]> = positions
            .chunks(self.config.position_batch_size)
            .collect();
        
        let mut all_evaluations = Vec::with_capacity(total_positions);
        let mut completed_count = 0;
        
        // Process batches sequentially but positions within batches in parallel
        for batch in batches {
            let batch_evaluations = self.evaluate_batch_internal(oracle, batch).await?;
            
            completed_count += batch.len();
            all_evaluations.extend(batch_evaluations);
            
            // Update progress
            if let Some(tracker) = &self.progress_tracker {
                tracker.update_oracle(completed_count, total_positions);
            }
            
            // Small delay to prevent overwhelming Stockfish
            sleep(Duration::from_millis(50)).await;
        }
        
        Ok(all_evaluations)
    }
    
    /// Internal batch evaluation with controlled concurrency
    async fn evaluate_batch_internal(
        &self,
        oracle: &mut StockfishOracle,
        positions: &[ChessPosition],
    ) -> Result<Vec<StockfishEvaluation>> {
        
        // For now, evaluate sequentially since Stockfish oracle is not thread-safe
        // In production, this could use multiple Stockfish processes
        let mut evaluations = Vec::with_capacity(positions.len());
        
        for position in positions {
            let evaluation = oracle.evaluate_position(position)?;
            evaluations.push(evaluation);
        }
        
        Ok(evaluations)
    }
}

/// Parallel Pattern Discovery Engine
/// Provides concurrent mathematical pattern discovery with progress tracking
pub struct ParallelPatternDiscovery {
    config: ParallelConfig,
    progress_tracker: Option<Arc<DiscoveryProgressTracker>>,
}

impl ParallelPatternDiscovery {
    /// Creates a new parallel pattern discovery engine
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            progress_tracker: None,
        }
    }
    
    /// Sets progress tracker for discovery monitoring
    pub fn with_progress_tracker(mut self, tracker: Arc<DiscoveryProgressTracker>) -> Self {
        self.progress_tracker = Some(tracker);
        self
    }
    
    /// Discovers patterns in parallel across position vectors
    /// Mathematical form: Π_parallel: ℝ^{n×m} → 𝒫(Patterns)
    pub fn discover_patterns_parallel(
        &self,
        discovery_engine: &mut ChessMathematicalDiscoveryEngine,
        positions: &[ChessPosition],
    ) -> Result<DiscoveryResults> {
        
        let total_positions = positions.len();
        
        // Update progress tracker
        if let Some(tracker) = &self.progress_tracker {
            tracker.update_patterns(0, 0);
        }
        
        // Convert positions to vectors in parallel
        let position_vectors: Vec<Array1<f64>> = positions
            .par_iter()
            .map(|pos| pos.to_vector())
            .collect();
        
        // Run discovery cycle
        let discovery_results = discovery_engine.run_discovery_cycle(positions)?;
        
        // Update progress with final counts
        if let Some(tracker) = &self.progress_tracker {
            tracker.update_patterns(total_positions, discovery_results.new_patterns.len());
        }
        
        Ok(discovery_results)
    }
    
    /// Parallel vectorization of chess positions
    /// Mathematical form: φ_parallel: {s₁, s₂, ..., sₙ} → {v₁, v₂, ..., vₙ} where vᵢ ∈ ℝ^1024
    pub fn vectorize_positions_parallel(&self, positions: &[ChessPosition]) -> Vec<Array1<f64>> {
        positions
            .par_iter()
            .map(|position| position.to_vector())
            .collect()
    }
    
    /// Parallel position generation for diverse sampling
    /// Mathematical form: Generates diverse subset of S with parallel processing
    pub fn generate_diverse_positions_parallel(&self, count: usize) -> Vec<ChessPosition> {
        (0..count)
            .into_par_iter()
            .map(|i| {
                match i % 4 {
                    0 => ChessPosition::generate_endgame_position(),
                    1 => ChessPosition::generate_tactical_position(),
                    2 => ChessPosition::generate_positional_position(),
                    _ => ChessPosition::generate_random_position(),
                }
            })
            .collect()
    }
}

/// Parallel Knowledge Base Analyzer
/// Provides concurrent knowledge base analysis and consistency checking
pub struct ParallelKnowledgeAnalyzer {
    config: ParallelConfig,
    progress_tracker: Option<Arc<DiscoveryProgressTracker>>,
}

impl ParallelKnowledgeAnalyzer {
    /// Creates a new parallel knowledge base analyzer
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            progress_tracker: None,
        }
    }
    
    /// Sets progress tracker for analysis monitoring
    pub fn with_progress_tracker(mut self, tracker: Arc<DiscoveryProgressTracker>) -> Self {
        self.progress_tracker = Some(tracker);
        self
    }
    
    /// Analyzes knowledge base consistency in parallel
    /// Mathematical form: Parallel consistency checking across pattern space
    pub fn analyze_consistency_parallel(
        &self,
        knowledge_calculator: &crate::knowledge_metrics::KnowledgeDistanceCalculator,
        knowledge_base: &crate::discovery_engine::MathematicalKnowledgeBase,
    ) -> Result<f64> {
        
        // Update progress tracker
        if let Some(tracker) = &self.progress_tracker {
            tracker.update_kb_consistency(0.0);
        }
        
        // Perform consistency analysis
        let consistency_score = knowledge_calculator.validate_kb_consistency(knowledge_base)?;
        
        // Update progress with final score
        if let Some(tracker) = &self.progress_tracker {
            tracker.update_kb_consistency(consistency_score);
        }
        
        Ok(consistency_score)
    }
}

/// Progress Animation Task
/// Provides smooth progress updates and animations
pub struct ProgressAnimator {
    update_interval: Duration,
    is_running: Arc<Mutex<bool>>,
}

impl ProgressAnimator {
    /// Creates a new progress animator
    pub fn new(update_interval_ms: u64) -> Self {
        Self {
            update_interval: Duration::from_millis(update_interval_ms),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Starts animated progress updates
    pub async fn start_animation(&self, tracker: Arc<DiscoveryProgressTracker>) {
        *self.is_running.lock().unwrap() = true;
        
        while *self.is_running.lock().unwrap() {
            // Progress animation logic could go here
            // For now, just maintain the progress bars
            sleep(self.update_interval).await;
        }
    }
    
    /// Stops progress animation
    pub fn stop(&self) {
        *self.is_running.lock().unwrap() = false;
    }
}

/// Parallel Discovery Coordinator
/// Coordinates all parallel operations with unified progress tracking
pub struct ParallelDiscoveryCoordinator {
    config: ParallelConfig,
    oracle_evaluator: ParallelOracleEvaluator,
    pattern_discovery: ParallelPatternDiscovery,
    knowledge_analyzer: ParallelKnowledgeAnalyzer,
    progress_animator: ProgressAnimator,
}

impl ParallelDiscoveryCoordinator {
    /// Creates a new parallel discovery coordinator
    pub fn new(config: ParallelConfig) -> Self {
        let oracle_evaluator = ParallelOracleEvaluator::new(config.clone());
        let pattern_discovery = ParallelPatternDiscovery::new(config.clone());
        let knowledge_analyzer = ParallelKnowledgeAnalyzer::new(config.clone());
        let progress_animator = ProgressAnimator::new(config.progress_update_ms);
        
        Self {
            config,
            oracle_evaluator,
            pattern_discovery,
            knowledge_analyzer,
            progress_animator,
        }
    }
    
    /// Initializes progress tracking for all components
    pub fn initialize_progress_tracking(
        &mut self,
        total_positions: usize,
        total_iterations: usize,
    ) -> Arc<DiscoveryProgressTracker> {
        
        let tracker = Arc::new(DiscoveryProgressTracker::new(total_positions, total_iterations));
        
        self.oracle_evaluator = self.oracle_evaluator.clone().with_progress_tracker(tracker.clone());
        self.pattern_discovery = self.pattern_discovery.clone().with_progress_tracker(tracker.clone());
        self.knowledge_analyzer = self.knowledge_analyzer.clone().with_progress_tracker(tracker.clone());
        
        tracker
    }
    
    /// Gets reference to oracle evaluator
    pub fn oracle_evaluator(&self) -> &ParallelOracleEvaluator {
        &self.oracle_evaluator
    }
    
    /// Gets reference to pattern discovery
    pub fn pattern_discovery(&self) -> &ParallelPatternDiscovery {
        &self.pattern_discovery
    }
    
    /// Gets reference to knowledge analyzer
    pub fn knowledge_analyzer(&self) -> &ParallelKnowledgeAnalyzer {
        &self.knowledge_analyzer
    }
}

impl Clone for ParallelOracleEvaluator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            progress_tracker: self.progress_tracker.clone(),
        }
    }
}

impl Clone for ParallelPatternDiscovery {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            progress_tracker: self.progress_tracker.clone(),
        }
    }
}

impl Clone for ParallelKnowledgeAnalyzer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            progress_tracker: self.progress_tracker.clone(),
        }
    }
}