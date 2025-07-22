/*
Real-Game Validation System for Chess Mathematical Discovery

This module validates discovered mathematical patterns against actual game outcomes
from master-level chess games to ensure strategic relevance.

Mathematical Framework:
- Pattern Validation: P(pattern_predicts_outcome) = correlation(pattern_value, game_result)
- Strategic Relevance: relevance_score = Σ win_probability_alignment_i
- Outcome Prediction: predict_outcome(pattern) → {win, draw, loss}
*/

use ndarray::Array1;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;
use std::time::SystemTime;
use crate::discovery_engine::{DiscoveredPattern, MathematicalConstant, MathematicalFunction};
use crate::ChessPosition;

/// Game phase classification for phase-aware validation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GamePhase {
    Opening,    // Moves 1-10
    Middlegame, // Moves 11-25
    Endgame,    // Moves 26+
}

/// Game outcome from actual chess games
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameOutcome {
    WhiteWins,
    BlackWins,
    Draw,
}

/// Chess game record with positions and outcome - enhanced with engine evaluations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChessGame {
    pub game_id: String,
    pub white_player: String,
    pub black_player: String,
    pub white_elo: Option<u32>,
    pub black_elo: Option<u32>,
    pub positions: Vec<ChessPosition>,
    pub outcome: GameOutcome,
    pub played_at: SystemTime,
    /// Engine evaluations in centipawns for each position (if available)
    pub engine_evaluations: Option<Vec<f32>>,
    /// Evaluation changes between positions (momentum/advantage shifts)
    pub evaluation_deltas: Option<Vec<f32>>,
}

/// Enhanced validation result supporting multiple prediction targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternValidationResult {
    pub pattern_name: String,
    pub games_tested: usize,
    pub prediction_accuracy: f64,
    pub strategic_relevance: f64,
    pub outcome_correlation: f64,
    pub confidence_interval: (f64, f64),
    pub significant: bool,
    // Phase-specific validation results
    pub opening_accuracy: f64,
    pub middlegame_accuracy: f64,
    pub endgame_accuracy: f64,
    pub best_phase: GamePhase,
    // Enhanced evaluation metrics
    pub engine_evaluation_correlation: Option<f64>,
    pub positional_advantage_correlation: Option<f64>,
    pub evaluation_delta_correlation: Option<f64>,
}

/// Real-game validation system
pub struct GameOutcomeValidator {
    /// Historical game database
    game_database: Vec<ChessGame>,
    
    /// Validation thresholds
    min_games_for_validation: usize,
    significance_threshold: f64,
    relevance_threshold: f64,
}

impl GameOutcomeValidator {
    pub fn new() -> Self {
        Self {
            game_database: Vec::new(),
            min_games_for_validation: 200,  // Increased for better statistical significance
            significance_threshold: 0.65,   // Slightly higher threshold for quality
            relevance_threshold: 0.60,      // Increased threshold
        }
    }
    
    /// Load games from PGN or database
    pub fn load_games(&mut self, games: Vec<ChessGame>) -> Result<()> {
        self.game_database.extend(games);
        println!("🎯 Loaded {} games for validation", self.game_database.len());
        Ok(())
    }
    
    /// Validate discovered patterns against game outcomes
    /// Mathematical form: correlation(pattern_value, game_outcome)
    pub fn validate_patterns(&self, patterns: &[DiscoveredPattern]) -> Result<Vec<PatternValidationResult>> {
        let mut results = Vec::new();
        
        if self.game_database.len() < self.min_games_for_validation {
            println!("⚠️  Insufficient games for validation: {} < {}", 
                     self.game_database.len(), self.min_games_for_validation);
            return Ok(results);
        }
        
        println!("🎯 Validating {} patterns against {} games", 
                 patterns.len(), self.game_database.len());
        
        for pattern in patterns {
            match pattern {
                DiscoveredPattern::Constant { name, value, .. } => {
                    results.push(self.validate_constant_pattern(name, *value)?);
                }
                DiscoveredPattern::LinearRelationship { coefficient, intercept, feature_names, .. } => {
                    results.push(self.validate_linear_relationship(*coefficient, *intercept, feature_names)?);
                }
                DiscoveredPattern::SymbolicExpression { expression, feature_names, .. } => {
                    results.push(self.validate_symbolic_expression(expression, feature_names)?);
                }
                _ => {
                    // Handle other pattern types as needed
                }
            }
        }
        
        Ok(results)
    }
    
    /// Validate a mathematical constant against game outcomes with phase awareness
    fn validate_constant_pattern(&self, name: &str, value: f64) -> Result<PatternValidationResult> {
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        let mut outcome_correlations = Vec::new();
        
        // Phase-specific tracking
        let mut opening_correct = 0;
        let mut opening_total = 0;
        let mut middlegame_correct = 0;
        let mut middlegame_total = 0;
        let mut endgame_correct = 0;
        let mut endgame_total = 0;
        
        for game in &self.game_database {
            if game.positions.is_empty() {
                continue;
            }
            
            // For each position in the game, calculate the constant's predictive power
            for (pos_idx, position) in game.positions.iter().enumerate() {
                let position_vector = position.to_vector();
                
                // Determine game phase based on position index and move number
                let phase = self.determine_game_phase(position, pos_idx);
                
                // Calculate pattern value for this position
                let pattern_value = self.calculate_pattern_value_for_constant(name, value, &position_vector);
                
                // Predict outcome based on pattern value
                let predicted_outcome = self.predict_outcome_from_value(pattern_value);
                
                // Check if prediction matches actual outcome
                let is_correct = self.prediction_matches_outcome(&predicted_outcome, &game.outcome);
                
                if is_correct {
                    correct_predictions += 1;
                }
                total_predictions += 1;
                
                // Track phase-specific accuracy
                match phase {
                    GamePhase::Opening => {
                        if is_correct { opening_correct += 1; }
                        opening_total += 1;
                    },
                    GamePhase::Middlegame => {
                        if is_correct { middlegame_correct += 1; }
                        middlegame_total += 1;
                    },
                    GamePhase::Endgame => {
                        if is_correct { endgame_correct += 1; }
                        endgame_total += 1;
                    },
                }
                
                // Calculate correlation with game outcome
                let outcome_value = self.outcome_to_value(&game.outcome);
                outcome_correlations.push((pattern_value, outcome_value));
            }
        }
        
        let prediction_accuracy = if total_predictions > 0 {
            correct_predictions as f64 / total_predictions as f64
        } else {
            0.0
        };
        
        let outcome_correlation = self.calculate_correlation(&outcome_correlations);
        let strategic_relevance = self.calculate_strategic_relevance(prediction_accuracy, outcome_correlation);
        
        let confidence_interval = self.calculate_confidence_interval(
            correct_predictions, total_predictions, 0.95
        );
        
        let significant = prediction_accuracy > self.significance_threshold && 
                         strategic_relevance > self.relevance_threshold;
        
        println!("🎯 Constant '{}': accuracy={:.3}, relevance={:.3}, significant={}", 
                 name, prediction_accuracy, strategic_relevance, significant);
        
        // Calculate phase-specific accuracies
        let opening_accuracy = if opening_total > 0 {
            opening_correct as f64 / opening_total as f64
        } else { 0.0 };
        
        let middlegame_accuracy = if middlegame_total > 0 {
            middlegame_correct as f64 / middlegame_total as f64
        } else { 0.0 };
        
        let endgame_accuracy = if endgame_total > 0 {
            endgame_correct as f64 / endgame_total as f64
        } else { 0.0 };
        
        // Determine best phase
        let best_phase = if opening_accuracy >= middlegame_accuracy && opening_accuracy >= endgame_accuracy {
            GamePhase::Opening
        } else if middlegame_accuracy >= endgame_accuracy {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        };

        Ok(PatternValidationResult {
            pattern_name: name.to_string(),
            games_tested: self.game_database.len(),
            prediction_accuracy,
            strategic_relevance,
            outcome_correlation,
            confidence_interval,
            significant,
            opening_accuracy,
            middlegame_accuracy,
            endgame_accuracy,
            best_phase,
            engine_evaluation_correlation: None,
            positional_advantage_correlation: None,
            evaluation_delta_correlation: None,
        })
    }
    
    /// Validate a linear relationship against game outcomes with phase awareness
    fn validate_linear_relationship(&self, coefficient: f64, intercept: f64, feature_names: &(String, String)) -> Result<PatternValidationResult> {
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        let mut outcome_correlations = Vec::new();
        
        // Phase-specific tracking
        let mut opening_correct = 0;
        let mut opening_total = 0;
        let mut middlegame_correct = 0;
        let mut middlegame_total = 0;
        let mut endgame_correct = 0;
        let mut endgame_total = 0;
        
        for game in &self.game_database {
            for (pos_idx, position) in game.positions.iter().enumerate() {
                let position_vector = position.to_vector();
                
                // Determine game phase
                let phase = self.determine_game_phase(position, pos_idx);
                
                // Calculate linear relationship value: y = mx + b
                let feature_x = self.extract_feature_value(&position_vector, &feature_names.0);
                let pattern_value = coefficient * feature_x + intercept;
                
                
                let predicted_outcome = self.predict_outcome_from_value(pattern_value);
                
                
                let is_correct = self.prediction_matches_outcome(&predicted_outcome, &game.outcome);
                if is_correct {
                    correct_predictions += 1;
                }
                total_predictions += 1;
                
                // Track phase-specific accuracy
                match phase {
                    GamePhase::Opening => {
                        if is_correct { opening_correct += 1; }
                        opening_total += 1;
                    },
                    GamePhase::Middlegame => {
                        if is_correct { middlegame_correct += 1; }
                        middlegame_total += 1;
                    },
                    GamePhase::Endgame => {
                        if is_correct { endgame_correct += 1; }
                        endgame_total += 1;
                    },
                }
                
                let outcome_value = self.outcome_to_value(&game.outcome);
                outcome_correlations.push((pattern_value, outcome_value));
            }
        }
        
        let prediction_accuracy = if total_predictions > 0 {
            correct_predictions as f64 / total_predictions as f64
        } else {
            0.0
        };
        
        let outcome_correlation = self.calculate_correlation(&outcome_correlations);
        let strategic_relevance = self.calculate_strategic_relevance(prediction_accuracy, outcome_correlation);
        
        let confidence_interval = self.calculate_confidence_interval(
            correct_predictions, total_predictions, 0.95
        );
        
        let significant = prediction_accuracy > self.significance_threshold && 
                         strategic_relevance > self.relevance_threshold;
        
        // Calculate phase-specific accuracies
        let opening_accuracy = if opening_total > 0 {
            opening_correct as f64 / opening_total as f64
        } else { 0.0 };
        
        let middlegame_accuracy = if middlegame_total > 0 {
            middlegame_correct as f64 / middlegame_total as f64
        } else { 0.0 };
        
        let endgame_accuracy = if endgame_total > 0 {
            endgame_correct as f64 / endgame_total as f64
        } else { 0.0 };
        
        // Determine best phase
        let best_phase = if opening_accuracy >= middlegame_accuracy && opening_accuracy >= endgame_accuracy {
            GamePhase::Opening
        } else if middlegame_accuracy >= endgame_accuracy {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        };
        
        let pattern_name = format!("linear_{}_{}", feature_names.0, feature_names.1);
        
        println!("🎯 Linear pattern '{}': accuracy={:.3}, relevance={:.3}, significant={}", 
                 pattern_name, prediction_accuracy, strategic_relevance, significant);
        
        Ok(PatternValidationResult {
            pattern_name,
            games_tested: self.game_database.len(),
            prediction_accuracy,
            strategic_relevance,
            outcome_correlation,
            confidence_interval,
            significant,
            opening_accuracy,
            middlegame_accuracy,
            endgame_accuracy,
            best_phase,
            engine_evaluation_correlation: None,
            positional_advantage_correlation: None,
            evaluation_delta_correlation: None,
        })
    }
    
    /// Validate a symbolic expression against game outcomes with phase awareness
    fn validate_symbolic_expression(&self, expression: &crate::symbolic_regression::Expression, feature_names: &(String, String)) -> Result<PatternValidationResult> {
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        let mut outcome_correlations = Vec::new();
        
        // Phase-specific tracking
        let mut opening_correct = 0;
        let mut opening_total = 0;
        let mut middlegame_correct = 0;
        let mut middlegame_total = 0;
        let mut endgame_correct = 0;
        let mut endgame_total = 0;
        
        for game in &self.game_database {
            for (pos_idx, position) in game.positions.iter().enumerate() {
                let position_vector = position.to_vector();
                
                // Determine game phase
                let phase = self.determine_game_phase(position, pos_idx);
                
                // FIX: Pass complete feature vector to symbolic expression, not just single feature
                let pattern_value = expression.evaluate(position_vector.as_slice().unwrap());
                
                if pattern_value.is_finite() {
                    let predicted_outcome = self.predict_outcome_from_value(pattern_value);
                    
                    let is_correct = self.prediction_matches_outcome(&predicted_outcome, &game.outcome);
                    if is_correct {
                        correct_predictions += 1;
                    }
                    total_predictions += 1;
                    
                    // Track phase-specific accuracy
                    match phase {
                        GamePhase::Opening => {
                            if is_correct { opening_correct += 1; }
                            opening_total += 1;
                        },
                        GamePhase::Middlegame => {
                            if is_correct { middlegame_correct += 1; }
                            middlegame_total += 1;
                        },
                        GamePhase::Endgame => {
                            if is_correct { endgame_correct += 1; }
                            endgame_total += 1;
                        },
                    }
                    
                    let outcome_value = self.outcome_to_value(&game.outcome);
                    outcome_correlations.push((pattern_value, outcome_value));
                }
            }
        }
        
        let prediction_accuracy = if total_predictions > 0 {
            correct_predictions as f64 / total_predictions as f64
        } else {
            0.0
        };
        
        let outcome_correlation = self.calculate_correlation(&outcome_correlations);
        let strategic_relevance = self.calculate_strategic_relevance(prediction_accuracy, outcome_correlation);
        
        let confidence_interval = self.calculate_confidence_interval(
            correct_predictions, total_predictions, 0.95
        );
        
        let significant = prediction_accuracy > self.significance_threshold && 
                         strategic_relevance > self.relevance_threshold;
        
        // Calculate phase-specific accuracies
        let opening_accuracy = if opening_total > 0 {
            opening_correct as f64 / opening_total as f64
        } else { 0.0 };
        
        let middlegame_accuracy = if middlegame_total > 0 {
            middlegame_correct as f64 / middlegame_total as f64
        } else { 0.0 };
        
        let endgame_accuracy = if endgame_total > 0 {
            endgame_correct as f64 / endgame_total as f64
        } else { 0.0 };
        
        // Determine best phase
        let best_phase = if opening_accuracy >= middlegame_accuracy && opening_accuracy >= endgame_accuracy {
            GamePhase::Opening
        } else if middlegame_accuracy >= endgame_accuracy {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        };
        
        let pattern_name = format!("symbolic_{}", expression.to_string().chars().take(20).collect::<String>());
        
        println!("🎯 Symbolic pattern '{}': accuracy={:.3}, relevance={:.3}, significant={}", 
                 pattern_name, prediction_accuracy, strategic_relevance, significant);
        
        Ok(PatternValidationResult {
            pattern_name,
            games_tested: self.game_database.len(),
            prediction_accuracy,
            strategic_relevance,
            outcome_correlation,
            confidence_interval,
            significant,
            opening_accuracy,
            middlegame_accuracy,
            endgame_accuracy,
            best_phase,
            engine_evaluation_correlation: None,
            positional_advantage_correlation: None,
            evaluation_delta_correlation: None,
        })
    }
    
    /// Calculate pattern value for a constant
    fn calculate_pattern_value_for_constant(&self, name: &str, value: f64, position_vector: &Array1<f64>) -> f64 {
        // For constants, we relate them to strategic features
        match name {
            name if name.contains("material") => {
                // Material-related constants
                let material_balance = position_vector[768];
                value * material_balance
            }
            name if name.contains("center") => {
                // Center control constants
                let center_control = position_vector[772];
                value * center_control
            }
            name if name.contains("development") => {
                // Development constants
                let development = position_vector[773];
                value * development
            }
            _ => {
                // Generic constant application
                value * position_vector[768] // Apply to material balance
            }
        }
    }
    
    /// Extract feature value by name
    fn extract_feature_value(&self, position_vector: &Array1<f64>, feature_name: &str) -> f64 {
        // Map feature names to vector indices
        match feature_name {
            name if name.contains("feature_768") => position_vector[768], // Material balance
            name if name.contains("feature_769") => position_vector[769], // Positional score
            name if name.contains("feature_770") => position_vector[770], // White king safety
            name if name.contains("feature_771") => position_vector[771], // Black king safety
            name if name.contains("feature_772") => position_vector[772], // Center control
            name if name.contains("feature_773") => position_vector[773], // Development
            name if name.contains("feature_774") => position_vector[774], // Pawn structure
            _ => {
                // Try to parse feature index from name
                if let Some(idx_str) = feature_name.strip_prefix("feature_") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx < position_vector.len() {
                            return position_vector[idx];
                        }
                    }
                }
                0.0
            }
        }
    }
    
    /// Predict game outcome from pattern value with optimized thresholds
    fn predict_outcome_from_value(&self, pattern_value: f64) -> GameOutcome {
        // Dynamic threshold optimization based on pattern value distribution
        let win_threshold = self.calculate_optimal_win_threshold();
        let draw_threshold = self.calculate_optimal_draw_threshold();
        
        if pattern_value > win_threshold {
            GameOutcome::WhiteWins
        } else if pattern_value < -win_threshold {
            GameOutcome::BlackWins
        } else if pattern_value.abs() <= draw_threshold {
            GameOutcome::Draw
        } else if pattern_value > 0.0 {
            GameOutcome::WhiteWins
        } else {
            GameOutcome::BlackWins
        }
    }
    
    /// Calculate optimal win threshold based on game database statistics
    fn calculate_optimal_win_threshold(&self) -> f64 {
        if self.game_database.is_empty() {
            return 0.15; // Default threshold
        }
        
        let white_wins = self.game_database.iter()
            .filter(|g| matches!(g.outcome, GameOutcome::WhiteWins))
            .count() as f64;
        let total_games = self.game_database.len() as f64;
        let white_win_rate = white_wins / total_games;
        
        // Adjust threshold based on actual win rates in the database
        // Higher win rates suggest stronger decisive games, use higher threshold
        0.05 + (white_win_rate - 0.33).abs() * 0.3
    }
    
    /// Calculate optimal draw threshold
    fn calculate_optimal_draw_threshold(&self) -> f64 {
        if self.game_database.is_empty() {
            return 0.08; // Default threshold
        }
        
        let draws = self.game_database.iter()
            .filter(|g| matches!(g.outcome, GameOutcome::Draw))
            .count() as f64;
        let total_games = self.game_database.len() as f64;
        let draw_rate = draws / total_games;
        
        // Lower draw rates suggest more decisive games, use smaller draw threshold
        0.02 + draw_rate * 0.2
    }
    
    /// Check if prediction matches actual outcome
    fn prediction_matches_outcome(&self, predicted: &GameOutcome, actual: &GameOutcome) -> bool {
        match (predicted, actual) {
            (GameOutcome::WhiteWins, GameOutcome::WhiteWins) => true,
            (GameOutcome::BlackWins, GameOutcome::BlackWins) => true,
            (GameOutcome::Draw, GameOutcome::Draw) => true,
            _ => false,
        }
    }
    
    /// Convert outcome to numerical value for correlation
    fn outcome_to_value(&self, outcome: &GameOutcome) -> f64 {
        match outcome {
            GameOutcome::WhiteWins => 1.0,
            GameOutcome::BlackWins => -1.0,
            GameOutcome::Draw => 0.0,
        }
    }
    
    /// Determine game phase based on position and move number
    fn determine_game_phase(&self, position: &ChessPosition, _position_index: usize) -> GamePhase {
        // Use move number from position
        let move_number = position.fullmove_number;
        
        if move_number <= 10 {
            GamePhase::Opening
        } else if move_number <= 25 {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        }
    }
    
    /// Calculate correlation coefficient
    fn calculate_correlation(&self, data: &[(f64, f64)]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = data.iter().map(|(_, y)| y * y).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    /// Calculate strategic relevance score
    fn calculate_strategic_relevance(&self, prediction_accuracy: f64, outcome_correlation: f64) -> f64 {
        // Weighted combination of accuracy and correlation
        0.6 * prediction_accuracy + 0.4 * outcome_correlation.abs()
    }
    
    /// Calculate confidence interval for prediction accuracy
    fn calculate_confidence_interval(&self, successes: usize, total: usize, confidence_level: f64) -> (f64, f64) {
        if total == 0 {
            return (0.0, 0.0);
        }
        
        let p = successes as f64 / total as f64;
        let z = if confidence_level >= 0.95 { 1.96 } else { 1.645 }; // 95% or 90% confidence
        let margin_of_error = z * (p * (1.0 - p) / total as f64).sqrt();
        
        ((p - margin_of_error).max(0.0), (p + margin_of_error).min(1.0))
    }
    
    /// Generate summary report of validation results
    pub fn generate_validation_report(&self, results: &[PatternValidationResult]) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("# Real-Game Validation Report\n"));
        report.push_str(&format!("**Games Analyzed:** {}\n", self.game_database.len()));
        report.push_str(&format!("**Patterns Tested:** {}\n\n", results.len()));
        
        let significant_patterns = results.iter().filter(|r| r.significant).count();
        report.push_str(&format!("**Significant Patterns:** {}/{}\n\n", significant_patterns, results.len()));
        
        report.push_str("## Pattern Validation Results\n");
        report.push_str("| Pattern | Accuracy | Relevance | Correlation | Significant |\n");
        report.push_str("|---------|----------|-----------|-------------|-------------|\n");
        
        for result in results {
            report.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {} |\n",
                result.pattern_name,
                result.prediction_accuracy,
                result.strategic_relevance,
                result.outcome_correlation,
                if result.significant { "✅" } else { "❌" }
            ));
        }
        
        report.push_str("\n## Strategic Insights\n");
        
        let high_accuracy_patterns: Vec<_> = results.iter()
            .filter(|r| r.prediction_accuracy > 0.7)
            .collect();
        
        if !high_accuracy_patterns.is_empty() {
            report.push_str("### High-Accuracy Patterns\n");
            for pattern in high_accuracy_patterns {
                report.push_str(&format!(
                    "- **{}**: {:.1}% accuracy, {:.1}% relevance\n",
                    pattern.pattern_name,
                    pattern.prediction_accuracy * 100.0,
                    pattern.strategic_relevance * 100.0
                ));
            }
        }
        
        report
    }
}

impl Default for GameOutcomeValidator {
    fn default() -> Self {
        Self::new()
    }
}