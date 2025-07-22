/*
Stockfish Oracle Integration: E*(s) Ground Truth Evaluation

This module provides the ground truth evaluation function E*: S → ℝ
using Stockfish as our mathematical oracle for chess position evaluation.

Mathematical Framework:
- Oracle Function: E*(s) = stockfish_evaluation(s, depth)
- Evaluation Consistency: stability(E*(s)) = 1 - (σ/μ) across repeated evaluations
- Batch Processing: E*({s₁, s₂, ..., sₙ}) → {e₁, e₂, ..., eₙ}
*/

use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::time::Duration;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::ChessPosition;

/// Stockfish Oracle Configuration
/// Mathematical form: Configuration parameters for E*(s) evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockfishOracleConfig {
    /// Search depth for evaluations (deeper = more accurate but slower)
    pub depth: u8,
    
    /// Evaluation timeout in milliseconds
    pub timeout_ms: u64,
    
    /// Number of threads for Stockfish
    pub threads: u8,
    
    /// Hash table size in MB
    pub hash_size_mb: u16,
    
    /// Whether to use multiple evaluations for stability checking
    pub stability_check: bool,
    
    /// Number of evaluations per position for stability analysis
    pub stability_samples: u8,
}

impl Default for StockfishOracleConfig {
    fn default() -> Self {
        Self {
            depth: 20,           // Deep enough for accurate evaluation
            timeout_ms: 5000,    // 5 second timeout per position
            threads: 1,          // Single threaded for consistency
            hash_size_mb: 128,   // Reasonable hash size
            stability_check: true,
            stability_samples: 3,
        }
    }
}

/// Stockfish Evaluation Result
/// Mathematical form: (evaluation, confidence, stability, metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockfishEvaluation {
    /// Evaluation in centipawns: E*(s) ∈ ℝ
    pub evaluation_cp: f64,
    
    /// Confidence score: confidence ∈ [0,1]
    pub confidence: f64,
    
    /// Stability across multiple evaluations: stability = 1 - (σ/μ)
    pub stability: f64,
    
    /// Search depth actually achieved
    pub actual_depth: u8,
    
    /// Time taken for evaluation in milliseconds
    pub evaluation_time_ms: u64,
    
    /// Best move found by Stockfish
    pub best_move: Option<String>,
    
    /// Principal variation (sequence of best moves)
    pub principal_variation: Vec<String>,
    
    /// Whether this evaluation is from cache
    pub from_cache: bool,
}

/// Stockfish Oracle: Ground Truth Provider for E*(s)
/// Mathematical form: Oracle that provides E*(s) = stockfish_evaluation(s)
pub struct StockfishOracle {
    config: StockfishOracleConfig,
    
    /// Evaluation cache: position_hash → evaluation
    /// Mathematical form: cache: S → ℝ for memoization
    evaluation_cache: HashMap<String, StockfishEvaluation>,
    
    /// Performance metrics
    cache_hits: u64,
    cache_misses: u64,
    total_evaluations: u64,
    total_evaluation_time_ms: u64,
}

impl StockfishOracle {
    /// Creates a new Stockfish Oracle with configuration
    pub fn new(config: StockfishOracleConfig) -> Result<Self> {
        // Test that Stockfish is available
        let test_output = Command::new("stockfish")
            .arg("--help")
            .output()
            .context("Failed to execute Stockfish. Please ensure it's installed and in PATH")?;
        
        if !test_output.status.success() {
            return Err(anyhow::anyhow!("Stockfish not properly installed or accessible"));
        }
        
        println!("✅ Stockfish Oracle initialized successfully");
        println!("   - Depth: {}", config.depth);
        println!("   - Timeout: {}ms", config.timeout_ms);
        println!("   - Threads: {}", config.threads);
        println!("   - Hash: {}MB", config.hash_size_mb);
        
        Ok(Self {
            config,
            evaluation_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            total_evaluations: 0,
            total_evaluation_time_ms: 0,
        })
    }
    
    /// Evaluates a single chess position: E*(s) → ℝ
    /// Mathematical form: E*(s) = stockfish_evaluation(s, depth)
    pub fn evaluate_position(&mut self, position: &ChessPosition) -> Result<StockfishEvaluation> {
        let position_hash = self.compute_position_hash(position);
        
        // Check cache first
        if let Some(cached_eval) = self.evaluation_cache.get(&position_hash) {
            self.cache_hits += 1;
            let mut result = cached_eval.clone();
            result.from_cache = true;
            return Ok(result);
        }
        
        self.cache_misses += 1;
        self.total_evaluations += 1;
        
        // Convert position to FEN notation
        let fen = self.position_to_fen(position)?;
        
        // Perform evaluation(s)
        let evaluation = if self.config.stability_check {
            self.evaluate_with_stability_check(&fen)?
        } else {
            self.evaluate_single(&fen)?
        };
        
        // Cache the result
        self.evaluation_cache.insert(position_hash, evaluation.clone());
        
        Ok(evaluation)
    }
    
    /// Batch evaluation of multiple positions: E*({s₁, s₂, ..., sₙ}) → {e₁, e₂, ..., eₙ}
    /// Mathematical form: Efficient batch processing for large datasets
    pub fn evaluate_batch(&mut self, positions: &[ChessPosition]) -> Result<Vec<StockfishEvaluation>> {
        let mut evaluations = Vec::with_capacity(positions.len());
        
        println!("🔍 Evaluating {} positions with Stockfish oracle...", positions.len());
        let start_time = std::time::Instant::now();
        
        for (i, position) in positions.iter().enumerate() {
            if i % 10 == 0 {
                println!("   Progress: {}/{} positions evaluated", i, positions.len());
            }
            
            let evaluation = self.evaluate_position(position)?;
            evaluations.push(evaluation);
        }
        
        let elapsed = start_time.elapsed();
        println!("✅ Batch evaluation completed in {:.2}s", elapsed.as_secs_f64());
        println!("   - Cache hits: {}", self.cache_hits);
        println!("   - Cache misses: {}", self.cache_misses);
        println!("   - Average time per position: {:.2}ms", 
                 elapsed.as_millis() as f64 / positions.len() as f64);
        
        Ok(evaluations)
    }
    
    /// Evaluates position with stability checking: stability = 1 - (σ/μ)
    /// Mathematical form: Measures consistency across multiple evaluations
    fn evaluate_with_stability_check(&self, fen: &str) -> Result<StockfishEvaluation> {
        let mut evaluations = Vec::new();
        let mut total_time = 0u64;
        
        // Run multiple evaluations
        for _ in 0..self.config.stability_samples {
            let single_eval = self.evaluate_single(fen)?;
            evaluations.push(single_eval.evaluation_cp);
            total_time += single_eval.evaluation_time_ms;
        }
        
        // Calculate statistics
        let mean = evaluations.iter().sum::<f64>() / evaluations.len() as f64;
        let variance = evaluations.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / evaluations.len() as f64;
        let std_dev = variance.sqrt();
        
        // Stability calculation: stability = 1 - (σ/μ)
        let stability = if mean.abs() > 0.001 {
            1.0 - (std_dev / mean.abs())
        } else {
            1.0 // Perfect stability if evaluation is near zero
        };
        
        // Use the first evaluation as representative
        let mut base_eval = self.evaluate_single(fen)?;
        base_eval.evaluation_cp = mean;
        base_eval.stability = stability.max(0.0).min(1.0);
        base_eval.confidence = stability; // Higher stability = higher confidence
        base_eval.evaluation_time_ms = total_time / self.config.stability_samples as u64;
        
        Ok(base_eval)
    }
    
    /// Single evaluation using Stockfish
    fn evaluate_single(&self, fen: &str) -> Result<StockfishEvaluation> {
        let eval_start = std::time::Instant::now();
        
        let mut child = Command::new("stockfish")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn Stockfish process")?;
        
        // Configure Stockfish
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "setoption name Threads value {}", self.config.threads)?;
        writeln!(stdin, "setoption name Hash value {}", self.config.hash_size_mb)?;
        writeln!(stdin, "position fen {}", fen)?;
        writeln!(stdin, "go depth {}", self.config.depth)?;
        
        // Read output
        let stdout = child.stdout.as_mut().unwrap();
        let reader = BufReader::new(stdout);
        
        let mut evaluation_cp = 0.0;
        let mut best_move = None;
        let mut principal_variation = Vec::new();
        let mut actual_depth = 0;
        
        for line in reader.lines() {
            let line = line?;
            
            if line.starts_with("info depth") {
                if let Some(score_pos) = line.find(" score cp ") {
                    let score_str = &line[score_pos + 10..];
                    if let Some(space_pos) = score_str.find(' ') {
                        if let Ok(score) = score_str[..space_pos].parse::<i32>() {
                            evaluation_cp = score as f64;
                        }
                    }
                }
                
                if let Some(depth_pos) = line.find(" depth ") {
                    let depth_str = &line[depth_pos + 7..];
                    if let Some(space_pos) = depth_str.find(' ') {
                        if let Ok(depth) = depth_str[..space_pos].parse::<u8>() {
                            actual_depth = depth;
                        }
                    }
                }
                
                if let Some(pv_pos) = line.find(" pv ") {
                    let pv_str = &line[pv_pos + 4..];
                    principal_variation = pv_str.split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                }
            }
            
            if line.starts_with("bestmove") {
                best_move = line.split_whitespace().nth(1).map(|s| s.to_string());
                break;
            }
        }
        
        // Wait for process to complete
        let _output = child.wait_with_output()?;
        
        let evaluation_time = eval_start.elapsed().as_millis() as u64;
        
        Ok(StockfishEvaluation {
            evaluation_cp,
            confidence: 0.95, // Default high confidence for single evaluation
            stability: 1.0,   // Default perfect stability for single evaluation
            actual_depth,
            evaluation_time_ms: evaluation_time,
            best_move,
            principal_variation,
            from_cache: false,
        })
    }
    
    /// Converts chess position to FEN notation
    fn position_to_fen(&self, position: &ChessPosition) -> Result<String> {
        // This is a simplified FEN conversion - in production, this would be more robust
        let mut fen = String::new();
        
        // Board representation
        for rank in (0..8).rev() {
            let mut empty_count = 0;
            for file in 0..8 {
                if let Some(piece) = position.get_piece_at(rank, file) {
                    if empty_count > 0 {
                        fen.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    fen.push(piece_to_fen_char(piece));
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        
        // Side to move
        fen.push(' ');
        fen.push(if position.is_white_to_move() { 'w' } else { 'b' });
        
        // Simplified: add basic castling, en passant, and move counts
        fen.push_str(" KQkq - 0 1");
        
        Ok(fen)
    }
    
    /// Computes hash for position caching
    fn compute_position_hash(&self, position: &ChessPosition) -> String {
        // Simple hash based on position representation
        format!("{:?}", position)
    }
    
    /// Gets oracle performance statistics
    pub fn get_performance_stats(&self) -> OraclePerformanceStats {
        OraclePerformanceStats {
            total_evaluations: self.total_evaluations,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            cache_hit_rate: if self.total_evaluations > 0 {
                self.cache_hits as f64 / self.total_evaluations as f64
            } else {
                0.0
            },
            average_evaluation_time_ms: if self.total_evaluations > 0 {
                self.total_evaluation_time_ms as f64 / self.total_evaluations as f64
            } else {
                0.0
            },
            cached_positions: self.evaluation_cache.len(),
        }
    }
    
    /// Clears evaluation cache
    pub fn clear_cache(&mut self) {
        self.evaluation_cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

/// Oracle Performance Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePerformanceStats {
    pub total_evaluations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub average_evaluation_time_ms: f64,
    pub cached_positions: usize,
}

/// Converts piece to FEN character
fn piece_to_fen_char(piece: &crate::Piece) -> char {
    use crate::{PieceType, Color};
    
    let base_char = match piece.piece_type {
        PieceType::Pawn => 'p',
        PieceType::Knight => 'n',
        PieceType::Bishop => 'b',
        PieceType::Rook => 'r',
        PieceType::Queen => 'q',
        PieceType::King => 'k',
    };
    
    match piece.color {
        Color::White => base_char.to_ascii_uppercase(),
        Color::Black => base_char,
    }
}

/// Validation functions for oracle consistency
impl StockfishOracle {
    /// Validates oracle consistency across multiple evaluations with progress tracking
    /// Mathematical form: Checks if E*(s) is stable and consistent
    pub fn validate_oracle_consistency(&mut self, test_positions: &[ChessPosition]) -> Result<f64> {
        use indicatif::{ProgressBar, ProgressStyle};
        
        let sample_size = test_positions.len().min(10); // Sample validation
        let mut consistency_scores = Vec::new();
        
        // Create progress bar for consistency validation
        let progress_bar = ProgressBar::new((sample_size * 2) as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("🔍 Oracle Consistency: [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} evaluations ({percent}%) ETA: {eta}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        progress_bar.set_message("Validating Stockfish consistency");
        
        for (i, position) in test_positions.iter().take(sample_size).enumerate() {
            progress_bar.set_message(format!("Testing position {} consistency", i + 1));
            
            // First evaluation
            let eval1 = self.evaluate_position(position)?;
            progress_bar.inc(1);
            
            // Clear cache and re-evaluate
            let position_hash = self.compute_position_hash(position);
            self.evaluation_cache.remove(&position_hash);
            
            // Second evaluation
            let eval2 = self.evaluate_position(position)?;
            progress_bar.inc(1);
            
            // Calculate consistency: how close are the two evaluations?
            let consistency = 1.0 - (eval1.evaluation_cp - eval2.evaluation_cp).abs() / 
                              (eval1.evaluation_cp.abs() + eval2.evaluation_cp.abs() + 1.0);
            
            consistency_scores.push(consistency);
        }
        
        let average_consistency = consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64;
        
        progress_bar.finish_with_message("Oracle consistency validation complete");
        
        println!("🔍 Oracle consistency validation results:");
        println!("   - Average consistency: {:.3}", average_consistency);
        println!("   - Sample size: {} positions", sample_size);
        println!("   - Consistency range: [{:.3}, {:.3}]", 
                 consistency_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                 consistency_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        
        Ok(average_consistency)
    }
}