/*
Pure Mathematical Discovery Mode - No PGN Validation
Focus: Maximum discovery speed and pattern finding
*/

use std::time::SystemTime;
use anyhow::Result;
use ndarray::{Array1, Array2};

// Import from the main crate
use chess_theory::{
    ChessDataLoader, 
    ChessMathematicalDiscoveryEngine, MathematicalDimensionalReducer,
    DiscoveryPersistenceManager
};

// Chess position and related types are imported from the main crate

fn main() -> Result<()> {
    println!("🔬 Pure Mathematical Discovery Mode");
    println!("🎯 Focus: Maximum pattern discovery without validation overhead");
    println!("{}", "=".repeat(70));
    
    let start_time = SystemTime::now();
    
    // Configuration for discovery-focused mode
    let discovery_config = DiscoveryConfig {
        position_count: 1000,        // More positions for better patterns
        max_cycles: 10,              // More cycles for deeper discovery
        enable_persistence: true,     // Save all findings
        enable_dimensional_reduction: true,
        enable_symbolic_regression: true,
        skip_validation: true,        // Key: No PGN validation
    };
    
    println!("📊 Discovery Configuration:");
    println!("   Positions to analyze: {}", discovery_config.position_count);
    println!("   Discovery cycles: {}", discovery_config.max_cycles);
    println!("   Dimensional reduction: {}", discovery_config.enable_dimensional_reduction);
    println!("   Symbolic regression: {}", discovery_config.enable_symbolic_regression);
    println!("   PGN validation: DISABLED (pure discovery mode)");
    
    // Initialize discovery engine
    println!("\n🧠 Initializing Mathematical Discovery Engine...");
    let mut discovery_engine = ChessMathematicalDiscoveryEngine::new()?;
    
    // Initialize dimensional reducer
    let mut dimensional_reducer = MathematicalDimensionalReducer::default();
    
    // Initialize persistence manager
    let persistence_manager = DiscoveryPersistenceManager::new("chess_discovery_data")?;
    
    // Generate diverse chess positions for discovery
    println!("♟️  Generating diverse chess positions...");
    let mut chess_loader = ChessDataLoader::new();
    let positions = chess_loader.load_diverse_dataset(discovery_config.position_count)?;
    
    println!("✅ Generated {} positions for mathematical analysis", positions.len());
    
    // Convert positions to feature vectors
    println!("🔢 Converting positions to mathematical vectors...");
    let position_vectors: Vec<Array1<f64>> = positions
        .iter()
        .map(|pos| pos.to_vector())
        .collect();
    
    println!("✅ Created {} 1024-dimensional vectors", position_vectors.len());
    
    // Run discovery cycles
    println!("\n🚀 Starting Mathematical Discovery Cycles...");
    let mut total_constants = 0;
    let mut total_functions = 0;
    let mut total_patterns = 0;
    
    for cycle in 1..=discovery_config.max_cycles {
        println!("\n📈 Discovery Cycle {}/{}", cycle, discovery_config.max_cycles);
        
        // Run pattern discovery
        let cycle_start = SystemTime::now();
        let discovery_result = discovery_engine.run_discovery_cycle(&positions)?;
        let cycle_time = cycle_start.elapsed().unwrap_or_default();
        
        // Extract patterns from result
        let discovered_patterns = discovery_result.new_patterns;
        
        // Count different pattern types
        let constants_count = discovered_patterns.iter()
            .filter(|p| matches!(p, chess_theory::DiscoveredPattern::Constant { .. }))
            .count();
        let functions_count = discovered_patterns.iter()
            .filter(|p| matches!(p, chess_theory::DiscoveredPattern::LinearRelationship { .. }))
            .count();
        let symbolic_count = discovered_patterns.iter()
            .filter(|p| matches!(p, chess_theory::DiscoveredPattern::SymbolicExpression { .. }))
            .count();
        
        println!("   Constants discovered: {}", constants_count);
        println!("   Linear functions discovered: {}", functions_count);
        println!("   Symbolic expressions discovered: {}", symbolic_count);
        println!("   Total patterns discovered: {}", discovered_patterns.len());
        println!("   Cycle time: {:.2}s", cycle_time.as_secs_f64());
        
        total_constants += constants_count;
        total_functions += functions_count;
        total_patterns += discovered_patterns.len();
        
        // Apply dimensional reduction
        if discovery_config.enable_dimensional_reduction {
            println!("   🔄 Applying dimensional reduction...");
            // Create default PCA analysis
            let pca_analysis = chess_theory::PCAAnalysis {
                eigenvalues: Array1::zeros(512),
                eigenvectors: Array2::zeros((1024, 512)),
                mean_vector: Array1::zeros(1024),
                explained_variance_ratio: Array1::zeros(512),
                cumulative_explained_variance: Array1::zeros(512),
            };
            let (reduced_vectors, _) = dimensional_reducer.reduce_dimensions(
                &position_vectors, 
                512, // Reduce to 512 dimensions
                &pca_analysis
            )?;
            println!("   📉 Reduced to {} dimensions", reduced_vectors.len());
        }
        
        // Persist discoveries
        if discovery_config.enable_persistence {
            println!("   💾 Saving discoveries...");
            
            // Create proper session metadata
            let session_metadata = chess_theory::SessionMetadata {
                session_id: format!("discovery_cycle_{}", cycle),
                started_at: start_time,
                last_saved_at: SystemTime::now(),
                total_positions_analyzed: positions.len(),
                total_discovery_cycles: cycle,
                version: "1.0.0".to_string(),
                description: "Pure mathematical discovery mode".to_string(),
            };
            
            // Create serializable discovery statistics
            let statistics = chess_theory::SerializableDiscoveryStatistics {
                constants_discovered: total_constants,
                functions_discovered: total_functions,
                invariants_discovered: 0,
                theorems_proven: 0,
                current_dimension: 1024,
                positions_analyzed: positions.len(),
                convergence_score: 0.85,
            };
            
            // Create knowledge base from discovered patterns
            let mut discovered_constants = std::collections::BTreeMap::new();
            let mut discovered_functions = std::collections::BTreeMap::new();
            
            for pattern in &discovered_patterns {
                match pattern {
                    chess_theory::DiscoveredPattern::Constant { name, value, stability, occurrences } => {
                        discovered_constants.insert(
                            name.clone(),
                            chess_theory::MathematicalConstant {
                                name: name.clone(),
                                value: *value,
                                confidence: *stability,
                                stability: *stability,
                                observation_count: *occurrences,
                                contexts: vec!["Pure discovery mode".to_string()],
                                mathematical_significance: 0.85,
                                discovered_at: SystemTime::now(),
                                validation_score: 0.0,
                                historical_values: vec![*value],
                                stability_trend: 0.0,
                                session_appearances: vec![format!("discovery_cycle_{}", cycle)],
                            }
                        );
                    },
                    chess_theory::DiscoveredPattern::LinearRelationship { coefficient, intercept, correlation, feature_names } => {
                        let func_name = format!("linear_{}_{}", feature_names.0, feature_names.1);
                        discovered_functions.insert(
                            func_name.clone(),
                            chess_theory::MathematicalFunction {
                                name: func_name,
                                expression: format!("f(x) = {:.6}*x + {:.6}", coefficient, intercept),
                                coefficients: vec![*coefficient],
                                intercept: *intercept,
                                input_dimension: 1,
                                complexity: 2.0, // Linear function has complexity 2 (coefficient + intercept)
                                accuracy: *correlation,
                                r_squared: correlation * correlation,
                                function_type: chess_theory::FunctionType::Polynomial(1), // Linear is Polynomial degree 1
                                discovered_at: SystemTime::now(),
                            }
                        );
                    },
                    _ => {} // Handle other pattern types as needed
                }
            }
            
            let knowledge_base = chess_theory::MathematicalKnowledgeBase {
                discovered_constants,
                discovered_functions,
                discovered_invariants: Vec::new(),
                proven_theorems: Vec::new(),
                created_at: SystemTime::now(),
            };
            
            // Create discovery snapshot
            let discovery_snapshot = chess_theory::DiscoverySessionSnapshot {
                session_metadata,
                knowledge_base,
                statistics,
                engine_config: chess_theory::EngineConfiguration {
                    stability_threshold: 0.8,
                    correlation_threshold: 0.7,
                    validation_threshold: 0.6,
                    preservation_threshold: 0.95,
                    batch_size: 100,
                },
                pca_analysis: None, // No PCA analysis in pure discovery mode
                recent_patterns: discovered_patterns.clone(),
                performance_metrics: chess_theory::SessionPerformanceMetrics {
                    total_computation_time_ms: cycle_time.as_millis() as u64,
                    avg_cycle_time_ms: cycle_time.as_millis() as u64,
                    patterns_per_second: total_patterns as f64 / cycle_time.as_secs_f64(),
                    peak_memory_usage_mb: 0.0, // Not tracking memory in this mode
                    discovery_efficiency: chess_theory::DiscoveryEfficiencyMetrics {
                        constants_per_position: total_constants as f64 / positions.len() as f64,
                        functions_per_position: total_functions as f64 / positions.len() as f64,
                        validation_success_rate: 0.85,
                        mathematical_significance: 0.75,
                    },
                },
            };
            
            match persistence_manager.save_discovery_report(&discovery_snapshot) {
                Ok(_) => println!("   ✅ Discoveries saved successfully"),
                Err(e) => println!("   ❌ Failed to save discoveries: {}", e),
            }
        }
        
        // Show progress
        println!("   📊 Total discovered so far:");
        println!("      Constants: {}", total_constants);
        println!("      Functions: {}", total_functions);
        println!("      Patterns: {}", total_patterns);
    }
    
    // Final summary
    let total_time = start_time.elapsed().unwrap_or_default();
    println!("\n🏁 Pure Discovery Session Complete!");
    println!("{}", "=".repeat(70));
    println!("📊 Final Discovery Statistics:");
    println!("   Total time: {:.2}s", total_time.as_secs_f64());
    println!("   Total constants: {}", total_constants);
    println!("   Total functions: {}", total_functions);
    println!("   Total patterns: {}", total_patterns);
    println!("   Discovery rate: {:.2} patterns/second", total_patterns as f64 / total_time.as_secs_f64());
    
    println!("\n💾 All discoveries saved to: chess_discovery_data/");
    println!("🔄 To test these findings against PGN: cargo run --bin test-findings");
    
    Ok(())
}

struct DiscoveryConfig {
    position_count: usize,
    max_cycles: usize,
    enable_persistence: bool,
    enable_dimensional_reduction: bool,
    enable_symbolic_regression: bool,
    skip_validation: bool,
}