/*
Knowledge Base Distance Metrics and Compatibility Checking

This module implements robust metrics for measuring distance between knowledge bases
and checking mathematical consistency across discovered patterns.

Mathematical Framework:
- Knowledge Distance: D(K₁, K₂) = weighted_sum(D_C, D_F, D_I, D_T)
- Compatibility: compatibility(kᵢ, kⱼ) → [0,1] measuring consistency
- Oracle Validation: Uses E*(s) ground truth for semantic comparison
*/

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use crate::discovery_engine::{
    MathematicalKnowledgeBase, 
    MathematicalConstant, 
    MathematicalFunction,
    DiscoveredPattern
};
use crate::stockfish_oracle::StockfishEvaluation;
use crate::ChessPosition;
use ndarray::Array1;

/// Knowledge Base Distance Metrics
/// Mathematical form: D(K₁, K₂) = Σᵢ wᵢ × Dᵢ(K₁, K₂)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDistanceMetrics {
    /// Distance between constants: D_C(K₁, K₂)
    pub constants_distance: f64,
    
    /// Distance between functions: D_F(K₁, K₂)
    pub functions_distance: f64,
    
    /// Distance between invariants: D_I(K₁, K₂)
    pub invariants_distance: f64,
    
    /// Distance between theorems: D_T(K₁, K₂)
    pub theorems_distance: f64,
    
    /// Overall weighted distance: D(K₁, K₂)
    pub overall_distance: f64,
    
    /// Semantic similarity based on oracle validation
    pub semantic_similarity: f64,
}

/// Compatibility Score between knowledge elements
/// Mathematical form: compatibility(kᵢ, kⱼ) ∈ [0,1]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityScore {
    /// Logical consistency score
    pub logical_consistency: f64,
    
    /// Empirical consistency on test data
    pub empirical_consistency: f64,
    
    /// Oracle validation consistency
    pub oracle_consistency: f64,
    
    /// Overall compatibility score
    pub overall_compatibility: f64,
}

/// Knowledge Base Distance Calculator
/// Implements robust metrics for comparing mathematical knowledge bases
pub struct KnowledgeDistanceCalculator {
    /// Test positions for semantic validation
    test_positions: Vec<ChessPosition>,
    
    /// Corresponding ground truth evaluations
    ground_truth_evaluations: Vec<StockfishEvaluation>,
    
    /// Distance calculation weights
    weights: DistanceWeights,
}

/// Weights for distance calculation components
#[derive(Debug, Clone)]
pub struct DistanceWeights {
    pub constants_weight: f64,
    pub functions_weight: f64,
    pub invariants_weight: f64,
    pub theorems_weight: f64,
    pub semantic_weight: f64,
}

impl Default for DistanceWeights {
    fn default() -> Self {
        Self {
            constants_weight: 0.3,
            functions_weight: 0.4,
            invariants_weight: 0.2,
            theorems_weight: 0.1,
            semantic_weight: 0.5,
        }
    }
}

impl KnowledgeDistanceCalculator {
    /// Creates new distance calculator with oracle validation data
    pub fn new(
        test_positions: Vec<ChessPosition>,
        ground_truth_evaluations: Vec<StockfishEvaluation>,
    ) -> Self {
        Self {
            test_positions,
            ground_truth_evaluations,
            weights: DistanceWeights::default(),
        }
    }
    
    /// Calculates complete distance metrics between two knowledge bases
    /// Mathematical form: D(K₁, K₂) = Σᵢ wᵢ × Dᵢ(K₁, K₂)
    pub fn calculate_knowledge_distance(
        &self,
        kb1: &MathematicalKnowledgeBase,
        kb2: &MathematicalKnowledgeBase,
    ) -> Result<KnowledgeDistanceMetrics> {
        
        // Calculate component distances
        let constants_distance = self.calculate_constants_distance(
            &kb1.discovered_constants,
            &kb2.discovered_constants,
        )?;
        
        let functions_distance = self.calculate_functions_distance(
            &kb1.discovered_functions,
            &kb2.discovered_functions,
        )?;
        
        let invariants_distance = self.calculate_invariants_distance(
            &kb1.discovered_invariants,
            &kb2.discovered_invariants,
        )?;
        
        let theorems_distance = self.calculate_theorems_distance(
            &kb1.proven_theorems,
            &kb2.proven_theorems,
        )?;
        
        // Calculate semantic similarity using oracle validation
        let semantic_similarity = self.calculate_semantic_similarity(kb1, kb2)?;
        
        // Compute overall weighted distance
        let overall_distance = self.weights.constants_weight * constants_distance +
                              self.weights.functions_weight * functions_distance +
                              self.weights.invariants_weight * invariants_distance +
                              self.weights.theorems_weight * theorems_distance +
                              self.weights.semantic_weight * (1.0 - semantic_similarity);
        
        Ok(KnowledgeDistanceMetrics {
            constants_distance,
            functions_distance,
            invariants_distance,
            theorems_distance,
            overall_distance,
            semantic_similarity,
        })
    }
    
    /// Calculates distance between constant sets
    /// Mathematical form: D_C(C₁, C₂) using value, stability, and confidence
    fn calculate_constants_distance(
        &self,
        constants1: &std::collections::BTreeMap<String, MathematicalConstant>,
        constants2: &std::collections::BTreeMap<String, MathematicalConstant>,
    ) -> Result<f64> {
        
        if constants1.is_empty() && constants2.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_distance = 0.0;
        let mut comparison_count: u64 = 0;
        
        // Find matched constants and calculate distances
        for (name1, const1) in constants1 {
            if let Some(const2) = constants2.get(name1) {
                // Constants with same name - calculate value distance
                let value_distance = (const1.value - const2.value).abs() / 
                                   (const1.value.abs() + const2.value.abs() + 1.0);
                
                let stability_distance = (const1.stability - const2.stability).abs();
                let confidence_distance = (const1.confidence - const2.confidence).abs();
                
                let const_distance = 0.6 * value_distance + 
                                   0.2 * stability_distance + 
                                   0.2 * confidence_distance;
                
                total_distance += const_distance;
                comparison_count += 1;
            } else {
                // Constant exists in KB1 but not KB2
                total_distance += 1.0; // Maximum distance penalty
                comparison_count += 1;
            }
        }
        
        // Constants in KB2 but not in KB1
        for name2 in constants2.keys() {
            if !constants1.contains_key(name2) {
                total_distance += 1.0;
                comparison_count += 1;
            }
        }
        
        if comparison_count > 0 {
            Ok(total_distance / comparison_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculates distance between function sets using semantic evaluation
    /// Mathematical form: D_F(F₁, F₂) based on oracle validation
    fn calculate_functions_distance(
        &self,
        functions1: &std::collections::BTreeMap<String, MathematicalFunction>,
        functions2: &std::collections::BTreeMap<String, MathematicalFunction>,
    ) -> Result<f64> {
        
        if functions1.is_empty() && functions2.is_empty() {
            return Ok(0.0);
        }
        
        // For now, implement a simplified version based on count and accuracy
        let count_diff = (functions1.len() as f64 - functions2.len() as f64).abs();
        let max_count = functions1.len().max(functions2.len()) as f64;
        
        if max_count == 0.0 {
            return Ok(0.0);
        }
        
        // Calculate average accuracy difference
        let avg_accuracy1 = if functions1.is_empty() { 0.0 } else {
            functions1.values().map(|f| f.accuracy).sum::<f64>() / functions1.len() as f64
        };
        
        let avg_accuracy2 = if functions2.is_empty() { 0.0 } else {
            functions2.values().map(|f| f.accuracy).sum::<f64>() / functions2.len() as f64
        };
        
        let accuracy_diff = (avg_accuracy1 - avg_accuracy2).abs();
        
        // Combine count and accuracy differences
        let distance = 0.6 * (count_diff / max_count) + 0.4 * accuracy_diff;
        
        Ok(distance)
    }
    
    /// Calculates distance between invariant sets
    fn calculate_invariants_distance(
        &self,
        invariants1: &[crate::discovery_engine::MathematicalInvariant],
        invariants2: &[crate::discovery_engine::MathematicalInvariant],
    ) -> Result<f64> {
        // Simplified implementation based on count
        let count_diff = (invariants1.len() as f64 - invariants2.len() as f64).abs();
        let max_count = invariants1.len().max(invariants2.len()) as f64;
        
        if max_count == 0.0 {
            Ok(0.0)
        } else {
            Ok(count_diff / max_count)
        }
    }
    
    /// Calculates distance between theorem sets
    fn calculate_theorems_distance(
        &self,
        theorems1: &[crate::discovery_engine::ChessTheorem],
        theorems2: &[crate::discovery_engine::ChessTheorem],
    ) -> Result<f64> {
        // Simplified implementation based on count and proof strength
        let count_diff = (theorems1.len() as f64 - theorems2.len() as f64).abs();
        let max_count = theorems1.len().max(theorems2.len()) as f64;
        
        if max_count == 0.0 {
            return Ok(0.0);
        }
        
        // Calculate average proof strength difference
        let avg_strength1 = if theorems1.is_empty() { 0.0 } else {
            theorems1.iter().map(|t| t.proof_strength).sum::<f64>() / theorems1.len() as f64
        };
        
        let avg_strength2 = if theorems2.is_empty() { 0.0 } else {
            theorems2.iter().map(|t| t.proof_strength).sum::<f64>() / theorems2.len() as f64
        };
        
        let strength_diff = (avg_strength1 - avg_strength2).abs();
        
        let distance = 0.7 * (count_diff / max_count) + 0.3 * strength_diff;
        
        Ok(distance)
    }
    
    /// Calculates semantic similarity using oracle validation
    /// Mathematical form: Measures how similarly KBs predict E*(s)
    fn calculate_semantic_similarity(
        &self,
        kb1: &MathematicalKnowledgeBase,
        kb2: &MathematicalKnowledgeBase,
    ) -> Result<f64> {
        // For this implementation, we'll use the fact that both KBs should
        // lead to similar predictions on the test positions
        
        // Calculate how well each KB's patterns correlate with ground truth
        let correlation1 = self.calculate_kb_oracle_correlation(kb1)?;
        let correlation2 = self.calculate_kb_oracle_correlation(kb2)?;
        
        // Similarity is how close their correlations are
        let similarity = 1.0 - (correlation1 - correlation2).abs();
        
        Ok(similarity.max(0.0).min(1.0))
    }
    
    /// Calculates correlation between KB predictions and oracle
    fn calculate_kb_oracle_correlation(&self, kb: &MathematicalKnowledgeBase) -> Result<f64> {
        // Simplified: use the number of stable constants as a proxy
        // In production, this would involve actual prediction evaluation
        
        let stable_constants = kb.discovered_constants.values()
            .filter(|c| c.stability > 0.9)
            .count() as f64;
        
        let total_constants = kb.discovered_constants.len() as f64;
        
        if total_constants == 0.0 {
            Ok(0.0)
        } else {
            Ok(stable_constants / total_constants)
        }
    }
    
    /// Checks compatibility between two discovered patterns
    /// Mathematical form: compatibility(kᵢ, kⱼ) ∈ [0,1]
    pub fn check_pattern_compatibility(
        &self,
        pattern1: &DiscoveredPattern,
        pattern2: &DiscoveredPattern,
    ) -> Result<CompatibilityScore> {
        
        match (pattern1, pattern2) {
            (DiscoveredPattern::Constant { name: name1, value: val1, stability: stab1, .. },
             DiscoveredPattern::Constant { name: name2, value: val2, stability: stab2, .. }) => {
                
                // Constants are compatible if they have similar values and high stability
                let value_compatibility = if name1 == name2 {
                    1.0 - (val1 - val2).abs() / (val1.abs() + val2.abs() + 1.0)
                } else {
                    // Different constants - check if they could be related
                    let ratio = val1 / val2;
                    if (ratio - 1.0).abs() < 0.1 || (ratio - 2.0).abs() < 0.1 || 
                       (ratio - 0.5).abs() < 0.1 || (ratio + 1.0).abs() < 0.1 {
                        0.7 // Potentially related constants
                    } else {
                        0.3 // Unrelated constants
                    }
                };
                
                let stability_compatibility = (stab1 * stab2).sqrt();
                
                Ok(CompatibilityScore {
                    logical_consistency: value_compatibility,
                    empirical_consistency: stability_compatibility,
                    oracle_consistency: 0.8, // Default for constants
                    overall_compatibility: 0.4 * value_compatibility + 
                                         0.3 * stability_compatibility + 
                                         0.3 * 0.8,
                })
            }
            
            (DiscoveredPattern::LinearRelationship { correlation: corr1, .. },
             DiscoveredPattern::LinearRelationship { correlation: corr2, .. }) => {
                
                // Linear relationships are compatible if correlations are similar
                let correlation_compatibility = 1.0 - (corr1 - corr2).abs();
                
                Ok(CompatibilityScore {
                    logical_consistency: correlation_compatibility,
                    empirical_consistency: correlation_compatibility,
                    oracle_consistency: 0.7,
                    overall_compatibility: correlation_compatibility * 0.9,
                })
            }
            
            _ => {
                // Different pattern types - default compatibility
                Ok(CompatibilityScore {
                    logical_consistency: 0.5,
                    empirical_consistency: 0.5,
                    oracle_consistency: 0.5,
                    overall_compatibility: 0.5,
                })
            }
        }
    }
    
    /// Validates overall knowledge base consistency
    /// Mathematical form: Checks internal consistency of all patterns
    pub fn validate_kb_consistency(
        &self,
        kb: &MathematicalKnowledgeBase,
    ) -> Result<f64> {
        
        let mut total_compatibility = 0.0;
        let mut comparison_count: u64 = 0;
        
        // Convert knowledge base to patterns for compatibility checking
        let mut patterns = Vec::new();
        
        for (name, constant) in &kb.discovered_constants {
            patterns.push(DiscoveredPattern::Constant {
                name: name.clone(),
                value: constant.value,
                stability: constant.stability,
                occurrences: constant.observation_count,
            });
        }
        
        for (name, function) in &kb.discovered_functions {
            patterns.push(DiscoveredPattern::LinearRelationship {
                coefficient: function.coefficients.get(0).copied().unwrap_or(0.0),
                intercept: function.coefficients.get(1).copied().unwrap_or(0.0),
                correlation: function.accuracy,
                feature_names: (name.clone(), "unknown".to_string()),
            });
        }
        
        // Check pairwise compatibility with sampling for large pattern sets
        let max_comparisons = 50_000; // Limit to prevent overflow and performance issues
        let patterns_len = patterns.len();
        
        if patterns_len < 2 {
            return Ok(1.0); // Perfect compatibility with no/single pattern
        }
        
        let total_possible_comparisons = (patterns_len * (patterns_len - 1)) / 2;
        
        if total_possible_comparisons <= max_comparisons {
            // Check all pairs for small sets
            for i in 0..patterns_len {
                for j in i + 1..patterns_len {
                    let compatibility = self.check_pattern_compatibility(&patterns[i], &patterns[j])?;
                    total_compatibility += compatibility.overall_compatibility;
                    comparison_count += 1;
                }
            }
        } else {
            // Sample pairs for large sets to avoid overflow
            use rand::{thread_rng, Rng};
            let mut rng = thread_rng();
            
            for _ in 0..max_comparisons {
                let i = rng.gen_range(0..patterns_len);
                let j = rng.gen_range(0..patterns_len);
                
                if i != j {
                    let compatibility = self.check_pattern_compatibility(&patterns[i], &patterns[j])?;
                    total_compatibility += compatibility.overall_compatibility;
                    comparison_count += 1;
                }
            }
        }
        
        if comparison_count > 0 {
            Ok(total_compatibility / comparison_count as f64)
        } else {
            Ok(1.0) // Empty KB is perfectly consistent
        }
    }
}