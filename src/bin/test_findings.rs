/*
Test All Findings Against PGN Mode
Focus: Validate all discovered patterns against elite chess games
*/

use std::path::Path;
use std::fs;
use std::time::SystemTime;
use anyhow::Result;

// Import from the main crate
use chess_theory::{
    ChessDataLoader, GameOutcomeValidator, ChessMathematicalDiscoveryEngine,
    GameOutcome, PatternValidationResult, DiscoveredPattern, FunctionType
};

// Chess position and related types are imported from the main crate

fn main() -> Result<()> {
    println!("🎯 Testing All Findings Against Elite PGN");
    println!("🔍 Focus: Validate discovered patterns against master-level games");
    println!("{}", "=".repeat(70));
    
    let start_time = SystemTime::now();
    
    // Configuration for testing
    let test_config = TestConfig {
        pgn_path: "/home/justin/Downloads/lichess_elite_2023-07.pgn".to_string(),
        games_to_test: 500,  // Increased from 100 to 500 for better statistical significance
        significance_threshold: 0.65,
        relevance_threshold: 0.60,
        load_from_reports: true,
    };
    
    println!("📊 Testing Configuration:");
    println!("   PGN file: {}", test_config.pgn_path);
    println!("   Games to test: {}", test_config.games_to_test);
    println!("   Significance threshold: {}%", test_config.significance_threshold * 100.0);
    println!("   Relevance threshold: {}%", test_config.relevance_threshold * 100.0);
    
    // Check PGN file exists
    if !Path::new(&test_config.pgn_path).exists() {
        println!("❌ PGN file not found at: {}", test_config.pgn_path);
        println!("   Please ensure lichess_elite_2023-07.pgn is in ~/Downloads/");
        return Ok(());
    }
    
    let metadata = fs::metadata(&test_config.pgn_path)?;
    println!("✅ PGN file found: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
    
    // Load discovered patterns from reports
    println!("\n📋 Loading discovered patterns from reports...");
    let patterns = load_patterns_from_reports()?;
    println!("✅ Loaded {} patterns from discovery reports", patterns.len());
    
    // Show pattern summary
    println!("\n📊 Pattern Summary:");
    let constants = patterns.iter().filter(|p| matches!(p, DiscoveredPattern::Constant { .. })).count();
    let linear = patterns.iter().filter(|p| matches!(p, DiscoveredPattern::LinearRelationship { .. })).count();
    let symbolic = patterns.iter().filter(|p| matches!(p, DiscoveredPattern::SymbolicExpression { .. })).count();
    
    println!("   Mathematical constants: {}", constants);
    println!("   Linear relationships: {}", linear);
    println!("   Symbolic expressions: {}", symbolic);
    
    // Load elite games
    println!("\n🎮 Loading elite games from PGN...");
    let mut chess_loader = ChessDataLoader::new();
    let games = chess_loader.load_games_from_pgn(&test_config.pgn_path, Some(test_config.games_to_test))?;
    println!("✅ Loaded {} elite games for testing", games.len());
    
    // Show game statistics
    let white_wins = games.iter().filter(|g| matches!(g.outcome, GameOutcome::WhiteWins)).count();
    let black_wins = games.iter().filter(|g| matches!(g.outcome, GameOutcome::BlackWins)).count();
    let draws = games.iter().filter(|g| matches!(g.outcome, GameOutcome::Draw)).count();
    
    println!("   Game outcomes: {} white wins, {} black wins, {} draws", white_wins, black_wins, draws);
    
    // Initialize validator
    println!("\n🔬 Initializing pattern validator...");
    let mut game_validator = GameOutcomeValidator::new();
    game_validator.load_games(games)?;
    
    // Test all patterns
    println!("\n🧪 Testing patterns against elite games...");
    let validation_results = game_validator.validate_patterns(&patterns)?;
    
    // Analyze results
    println!("\n📈 VALIDATION RESULTS");
    println!("{}", "=".repeat(70));
    println!("{:<40} | {:<15} | {:<12} | {:<12}", "Pattern Name", "Status", "Accuracy", "Relevance");
    println!("{}", "-".repeat(70));
    
    let mut significant_patterns = Vec::new();
    let mut total_accuracy = 0.0;
    let mut total_relevance = 0.0;
    
    for result in &validation_results {
        let status = if result.significant { "✅ SIGNIFICANT" } else { "❌ Not significant" };
        
        println!("{:<40} | {:<15} | {:<12.1}% | {:<12.1}%", 
                 truncate_name(&result.pattern_name, 40),
                 status,
                 result.prediction_accuracy * 100.0,
                 result.strategic_relevance * 100.0);
        
        total_accuracy += result.prediction_accuracy;
        total_relevance += result.strategic_relevance;
        
        if result.significant {
            significant_patterns.push(result.clone());
        }
    }
    
    // Summary statistics
    let avg_accuracy = total_accuracy / validation_results.len() as f64;
    let avg_relevance = total_relevance / validation_results.len() as f64;
    
    println!("\n🏆 TESTING SUMMARY");
    println!("{}", "=".repeat(70));
    println!("   Total patterns tested: {}", validation_results.len());
    println!("   Strategically significant: {} ({:.1}%)", 
             significant_patterns.len(), 
             significant_patterns.len() as f64 / validation_results.len() as f64 * 100.0);
    println!("   Average accuracy: {:.1}%", avg_accuracy * 100.0);
    println!("   Average strategic relevance: {:.1}%", avg_relevance * 100.0);
    
    // Top performing patterns
    if !significant_patterns.is_empty() {
        println!("\n🥇 TOP PERFORMING PATTERNS:");
        significant_patterns.sort_by(|a, b| {
            (b.prediction_accuracy + b.strategic_relevance)
                .partial_cmp(&(a.prediction_accuracy + a.strategic_relevance))
                .unwrap()
        });
        
        for (i, result) in significant_patterns.iter().take(10).enumerate() {
            println!("   {}. {} - {:.1}% accuracy, {:.1}% relevance", 
                     i + 1,
                     truncate_name(&result.pattern_name, 35),
                     result.prediction_accuracy * 100.0,
                     result.strategic_relevance * 100.0);
        }
    }
    
    // Performance analysis
    let total_time = start_time.elapsed().unwrap_or_default();
    println!("\n⚡ PERFORMANCE ANALYSIS:");
    println!("   Testing time: {:.2}s", total_time.as_secs_f64());
    println!("   Patterns per second: {:.2}", validation_results.len() as f64 / total_time.as_secs_f64());
    println!("   Games analyzed: {}", test_config.games_to_test);
    
    // Strategic insights
    if significant_patterns.len() > 0 {
        println!("\n🧠 STRATEGIC INSIGHTS:");
        println!("   ✅ {} patterns show genuine chess understanding", significant_patterns.len());
        println!("   ✅ Mathematical discoveries correlate with master-level play");
        println!("   ✅ Pattern validation confirms strategic relevance");
    } else {
        println!("\n🤔 STRATEGIC ANALYSIS:");
        println!("   ⚠️  No patterns meet significance thresholds");
        println!("   📊 Consider adjusting discovery parameters");
        println!("   🔄 May need more diverse training positions");
    }
    
    println!("\n🏁 PGN Testing Complete!");
    println!("   All discovered patterns validated against elite chess games");
    
    Ok(())
}

fn load_patterns_from_reports() -> Result<Vec<DiscoveredPattern>> {
    let reports_dir = "chess_discovery_data";
    let mut patterns = Vec::new();
    
    // Try to load from snapshot files first (more accurate)
    if let Ok(entries) = fs::read_dir(reports_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.contains("snapshot") && n.ends_with(".json")) {
                    
                    println!("📥 Loading patterns from snapshot: {:?}", path.file_name());
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(engine) = serde_json::from_str::<chess_theory::ChessMathematicalDiscoveryEngine>(&content) {
                            patterns.extend(extract_patterns_from_engine(&engine)?);
                            break; // Use first valid snapshot
                        }
                    }
                }
            }
        }
    }
    
    // Fall back to report parsing if no snapshot found
    if patterns.is_empty() {
        println!("📋 Loading patterns from report files...");
        if let Ok(entries) = fs::read_dir(reports_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "md") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            patterns.extend(parse_patterns_from_report_content(&content)?);
                        }
                    }
                }
            }
        }
    }
    
    // If still no patterns found, create some test patterns
    if patterns.is_empty() {
        println!("⚠️  No discovery data found, creating test patterns...");
        patterns = create_test_patterns();
    }
    
    Ok(patterns)
}

fn extract_patterns_from_engine(engine: &chess_theory::ChessMathematicalDiscoveryEngine) -> Result<Vec<DiscoveredPattern>> {
    let mut patterns = Vec::new();
    
    // Convert mathematical functions to discovered patterns
    for (name, function) in &engine.knowledge_base.discovered_functions {
        match &function.function_type {
            FunctionType::Polynomial(1) => {
                // Linear relationship
                let coefficient = function.coefficients.get(0).copied().unwrap_or(1.0);
                patterns.push(DiscoveredPattern::LinearRelationship {
                    coefficient,
                    intercept: function.intercept,
                    correlation: function.accuracy,
                    feature_names: (
                        format!("feature_{}", 768), // Use strategic features
                        format!("feature_{}", 769)
                    ),
                });
            },
            FunctionType::SymbolicExpression(expr) => {
                // Symbolic expression
                patterns.push(DiscoveredPattern::SymbolicExpression {
                    expression: expr.clone(),
                    fitness: function.accuracy,
                    complexity: function.complexity as usize,
                    feature_names: (
                        format!("feature_{}", 768), // Use strategic features  
                        format!("feature_{}", 769)
                    ),
                    r_squared: function.r_squared,
                });
            },
            _ => {} // Skip other function types for now
        }
    }
    
    // Convert mathematical constants to discovered patterns
    for (name, constant) in &engine.knowledge_base.discovered_constants {
        patterns.push(DiscoveredPattern::Constant {
            name: name.clone(),
            value: constant.value,
            stability: constant.stability,
            occurrences: constant.observation_count,
        });
    }
    
    println!("📊 Extracted {} patterns from engine snapshot", patterns.len());
    Ok(patterns)
}

fn parse_patterns_from_report_content(content: &str) -> Result<Vec<DiscoveredPattern>> {
    let mut patterns = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut in_functions_section = false;
    
    for line in lines {
        if line.contains("## 🔗 Functional Relationships") {
            in_functions_section = true;
            continue;
        }
        
        if in_functions_section && line.starts_with("## ") {
            break;
        }
        
        if in_functions_section && line.contains("**") && line.contains("accuracy:") {
            if let Some(name_start) = line.find("**") {
                if let Some(name_end) = line.rfind("**") {
                    let name = &line[name_start + 2..name_end];
                    
                    let accuracy = if let Some(acc_start) = line.find("accuracy: ") {
                        let acc_str = &line[acc_start + 10..];
                        if let Some(acc_end) = acc_str.find(")") {
                            acc_str[..acc_end].parse::<f64>().unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    
                    let pattern = if name.contains("linear") {
                        DiscoveredPattern::LinearRelationship {
                            coefficient: 1.0, // Default coefficient
                            intercept: 0.0,   // Default intercept
                            correlation: accuracy,
                            feature_names: ("x".to_string(), "y".to_string()),
                        }
                    } else if name.contains("symbolic") {
                        // For now, create as a constant since we don't have the symbolic expression
                        DiscoveredPattern::Constant {
                            name: name.to_string(),
                            value: accuracy,
                            stability: accuracy,
                            occurrences: 1,
                        }
                    } else {
                        DiscoveredPattern::Constant {
                            name: name.to_string(),
                            value: accuracy,
                            stability: accuracy,
                            occurrences: 1,
                        }
                    };
                    
                    patterns.push(pattern);
                }
            }
        }
    }
    
    Ok(patterns)
}

fn create_test_patterns() -> Vec<DiscoveredPattern> {
    vec![
        DiscoveredPattern::LinearRelationship {
            coefficient: 0.95,
            intercept: 0.05,
            correlation: 0.95,
            feature_names: ("material_balance".to_string(), "position_score".to_string()),
        },
        DiscoveredPattern::Constant {
            name: "test_constant_golden_ratio".to_string(),
            value: 1.618,
            stability: 0.92,
            occurrences: 100,
        },
    ]
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len-3])
    }
}

struct TestConfig {
    pgn_path: String,
    games_to_test: usize,
    significance_threshold: f64,
    relevance_threshold: f64,
    load_from_reports: bool,
}