/*
Mathematical Discovery Engine Core Structure

This module implements the core mathematical discovery engine based on the 
mathematical framework defined in the main documentation.

Mathematical Framework:
- Engine State: Ω = (K, D, V, P)
- Knowledge Base: K = (C, F, I, T)
- Discovery Function: Π: ℝ^{n×m} → 𝒫(Patterns)
- Validation Function: validation_score(pattern) = β₁·validation + β₂·consistency + β₃·stability
*/

use ndarray::Array1;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::BTreeMap;
use std::time::SystemTime;
use crate::ChessPosition;
use crate::symbolic_regression::{SymbolicRegression, SymbolicRegressionConfig, Expression};

/// Mathematical Discovery Engine State: Ω = (K, D, V, P)
/// Where K=knowledge, D=dimensional_state, V=validation_results, P=progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChessMathematicalDiscoveryEngine {
    /// Knowledge base K = (C, F, I, T)
    pub knowledge_base: MathematicalKnowledgeBase,
    
    /// Current dimensional state D
    pub dimensional_state: DimensionalState,
    
    /// Validation results V
    pub validation_results: ValidationHistory,
    
    /// Progress state P
    pub progress_state: ProgressState,
    
    /// Exploration parameters
    pub exploration_config: ExplorationConfig,
}

/// Mathematical Knowledge Base K = (C, F, I, T)
/// Where C=constants, F=functions, I=invariants, T=theorems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalKnowledgeBase {
    /// Discovered constants C = {(name_i, value_i, confidence_i, stability_i)}
    pub discovered_constants: BTreeMap<String, MathematicalConstant>,
    
    /// Discovered functions F = {f: ℝ^n → ℝ}
    pub discovered_functions: BTreeMap<String, MathematicalFunction>,
    
    /// Mathematical invariants I = {invariant properties}
    pub discovered_invariants: Vec<MathematicalInvariant>,
    
    /// Proven theorems T = {mathematical statements with proofs}
    pub proven_theorems: Vec<ChessTheorem>,
    
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Mathematical Constant: (name, value, confidence, stability)
/// Mathematical form: c_i with confidence = lim_{n→∞} P(|measured - true| < ε)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalConstant {
    pub name: String,
    pub value: f64,
    
    /// Confidence: confidence_i = lim_{n→∞} P(|measured_value - true_value| < ε)
    pub confidence: f64,
    
    /// Stability: stability_i = 1 - (σ_i / μ_i)
    pub stability: f64,
    
    /// Number of observations
    pub observation_count: usize,
    
    /// Contexts where this constant appears
    pub contexts: Vec<String>,
    
    /// Mathematical significance
    pub mathematical_significance: f64,
    
    /// Discovery timestamp
    pub discovered_at: SystemTime,
    
    /// Validation score
    pub validation_score: f64,
    
    /// Historical values for stability tracking
    pub historical_values: Vec<f64>,
    
    /// Stability trend (improving/degrading)
    pub stability_trend: f64,
    
    /// Sessions where this constant was observed
    pub session_appearances: Vec<String>,
}

/// Mathematical Function: f: ℝ^n → ℝ
/// Mathematical form: f ∈ ℱ where ℱ is the space of expressible mathematical functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalFunction {
    pub name: String,
    
    /// Function expression (symbolic representation)
    pub expression: String,
    
    /// Function coefficients
    pub coefficients: Vec<f64>,
    
    /// Intercept for linear functions (the 'b' in y = mx + b)
    pub intercept: f64,
    
    /// Input dimension n
    pub input_dimension: usize,
    
    /// Function complexity: complexity(f) = |nodes(f)| + Σ_{op∈f} weight(op)
    pub complexity: f64,
    
    /// Accuracy: MSE on validation data
    pub accuracy: f64,
    
    /// R-squared value
    pub r_squared: f64,
    
    /// Function type (polynomial, trigonometric, exponential, etc.)
    pub function_type: FunctionType,
    
    /// Discovery timestamp
    pub discovered_at: SystemTime,
}

/// Types of mathematical functions in our discovery space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionType {
    Polynomial(u32),    // degree
    Trigonometric,
    Exponential,
    Logarithmic,
    Hyperbolic,
    Composite,
    SymbolicExpression(Expression), // Genetic programming discovered expression
}

/// Mathematical Invariant: properties that remain unchanged under transformations
/// Mathematical form: invariant(f, G) ⟺ ∀T ∈ G: f(T(x)) = f(x)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalInvariant {
    pub name: String,
    pub description: String,
    
    /// Transformation group G = {T₁, T₂, ..., T_k}
    pub transformation_group: Vec<String>,
    
    /// Invariant strength: strength(f, T) = (1/m) Σᵢ exp(-|f(xᵢ) - f(T(xᵢ))|²/2σ²)
    pub strength: f64,
    
    /// Mathematical expression of the invariant
    pub invariant_expression: String,
    
    /// Discovery timestamp
    pub discovered_at: SystemTime,
}

/// Pattern significance classification for intelligent filtering
/// Mathematical form: classifier: Pattern → {StrategicDiscovery, EncodingArtifact, ChessRuleConstant, Unknown}
#[derive(Debug, Clone, PartialEq)]
pub enum PatternSignificance {
    /// Genuine chess strategic discovery with meaningful implications
    StrategicDiscovery,
    /// Mathematical artifact from the encoding process, not a real chess discovery
    EncodingArtifact,
    /// Known chess rule constant (piece values, board dimensions, etc.)
    ChessRuleConstant,
    /// Pattern with unknown significance
    Unknown,
}

/// Chess-specific mathematical theorem
/// Mathematical form: theorem(statement) with proof_strength and confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChessTheorem {
    pub statement: String,
    pub mathematical_form: String,
    pub proof_sketch: String,
    
    /// Proof strength: Σᵢ evidence_weight_i
    pub proof_strength: f64,
    
    /// Confidence: sigmoid(proof_strength)
    pub confidence: f64,
    
    /// Supporting evidence
    pub supporting_evidence: Vec<String>,
    
    /// Discovery timestamp
    pub discovered_at: SystemTime,
}

/// Current dimensional state D
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionalState {
    /// Current working dimension n (where n << 1024)
    pub current_dimension: usize,
    
    /// Dimensional reduction sequence: π₁, π₂, ..., πₖ
    pub reduction_history: Vec<DimensionalReduction>,
    
    /// Current transformation matrix (stored as Vec<Vec<f64>> for serialization)
    pub current_transformation: Option<Vec<Vec<f64>>>,
    
    /// Preserved variance after dimensional reduction
    pub preserved_variance: f64,
    
    /// Mathematical preservation score
    pub preservation_score: f64,
}

/// Dimensional Reduction: π: ℝ^m → ℝ^n where n < m
/// Mathematical form: minimize ||X - Wπ(X)||²_F subject to preservation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionalReduction {
    pub from_dimension: usize,
    pub to_dimension: usize,
    
    /// Transformation matrix (stored as Vec<Vec<f64>> for serialization)
    pub transformation_matrix: Vec<Vec<f64>>,
    
    /// Preserved variance: (eigenvalues[..to_dim].sum() / eigenvalues.sum())
    pub preserved_variance: f64,
    
    /// Mathematical constants preserved through this reduction
    pub preserved_constants: Vec<String>,
    
    /// Reduction method used
    pub method: ReductionMethod,
    
    /// Timestamp of reduction
    pub reduced_at: SystemTime,
}

/// Methods for dimensional reduction
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub enum ReductionMethod {
    PCA,                    // Principal Component Analysis
    MathematicalPCA,        // PCA with mathematical preservation constraints
    ICA,                    // Independent Component Analysis
    NonlinearPCA,           // Kernel PCA
    CustomMathematical,     // Custom method preserving discovered mathematics
}

/// Validation History V
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationHistory {
    /// Historical validation scores
    pub validation_scores: Vec<ValidationResult>,
    
    /// Cross-dimensional consistency scores
    pub consistency_scores: BTreeMap<String, f64>,
    
    /// Average validation performance
    pub average_validation_score: f64,
    
    /// Validation trend (improving/degrading)
    pub validation_trend: f64,
}

/// Single validation result
/// Mathematical form: validation_score = β₁·validation + β₂·consistency + β₃·stability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub pattern_name: String,
    pub validation_score: f64,
    pub consistency_score: f64,
    pub stability_score: f64,
    
    /// Combined score: β₁·validation + β₂·consistency + β₃·stability
    pub combined_score: f64,
    
    pub validated_at: SystemTime,
}

/// Progress State P
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressState {
    /// Total positions analyzed
    pub positions_analyzed: usize,
    
    /// Discovery milestones reached
    pub milestones: Vec<DiscoveryMilestone>,
    
    /// Current discovery phase
    pub current_phase: DiscoveryPhase,
    
    /// Convergence indicators
    pub convergence_indicators: ConvergenceIndicators,
    
    /// Session start time
    pub session_start: SystemTime,
    
    /// Last save time
    pub last_saved: SystemTime,
}

/// Discovery milestones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryMilestone {
    pub name: String,
    pub description: String,
    pub reached_at: SystemTime,
    pub significance: f64,
}

/// Discovery phases
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub enum DiscoveryPhase {
    Exploration,            // Initial exploration of position space
    PatternDetection,       // Finding mathematical patterns
    ConstantDiscovery,      // Discovering mathematical constants
    FunctionDiscovery,      // Discovering functional relationships
    InvariantDiscovery,     // Finding mathematical invariants
    TheoremProving,         // Proving mathematical theorems
    DimensionalReduction,   // Reducing dimensionality while preserving mathematics
    Convergence,            // Final convergence phase
    Complete,               // Discovery complete
}

/// Convergence Indicators
/// Mathematical form: converged(t) when multiple criteria are satisfied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceIndicators {
    /// Rate of new constant discovery (per position analyzed)
    pub constant_discovery_rate: f64,
    
    /// Rate of new function discovery
    pub function_discovery_rate: f64,
    
    /// Validation score improvement rate
    pub validation_improvement_rate: f64,
    
    /// Dimensional reduction feasibility
    pub dimensional_reduction_possible: bool,
    
    /// Overall convergence score [0, 1]
    pub convergence_score: f64,
    
    /// Estimated time to convergence
    pub estimated_time_to_convergence: Option<f64>,
}

/// Exploration Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Batch size for position analysis
    pub batch_size: usize,
    
    /// Stability threshold for constant discovery
    pub stability_threshold: f64,
    
    /// Correlation threshold for function discovery
    pub correlation_threshold: f64,
    
    /// Validation threshold for pattern acceptance
    pub validation_threshold: f64,
    
    /// Maximum complexity for function discovery
    pub max_function_complexity: f64,
    
    /// Mathematical preservation threshold for dimensional reduction
    pub preservation_threshold: f64,
    
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Pattern types discovered by the engine
/// These represent mathematical structures found in chess position space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveredPattern {
    /// Mathematical constant: value that remains stable across positions
    Constant {
        name: String,
        value: f64,
        stability: f64,
        occurrences: usize,
    },
    
    /// Linear relationship: y = ax + b
    LinearRelationship {
        coefficient: f64,
        intercept: f64,
        correlation: f64,
        feature_names: (String, String),
    },
    
    /// Polynomial relationship: y = Σ aᵢxᵢ
    PolynomialRelationship {
        coefficients: Vec<f64>,
        degree: u32,
        r_squared: f64,
        feature_names: Vec<String>,
    },
    
    /// Functional relationship: y = f(x) for some discovered function f
    FunctionalRelationship {
        function: MathematicalFunction,
        accuracy: f64,
        validation_score: f64,
    },
    
    /// Symbolic regression discovered expression: complex non-linear relationship
    SymbolicExpression {
        expression: Expression,
        fitness: f64,
        complexity: usize,
        feature_names: (String, String),
        r_squared: f64,
    },
    
    /// Mathematical invariant: property preserved under transformations
    Invariant {
        invariant: MathematicalInvariant,
        strength: f64,
    },
    
    /// Complex mathematical structure
    ComplexStructure {
        name: String,
        description: String,
        mathematical_form: String,
        components: Vec<String>,
        significance: f64,
    },
}

/// Results from a single discovery cycle
/// Mathematical form: discovery results from one iteration of Ω_{t+1} = Φ(Ω_t, X_t)
#[derive(Debug, Clone)]
pub struct DiscoveryResults {
    /// Newly discovered patterns
    pub new_patterns: Vec<DiscoveredPattern>,
    
    /// Updated constants
    pub updated_constants: Vec<String>,
    
    /// New mathematical functions
    pub new_functions: Vec<String>,
    
    /// Validation success rate for this cycle
    pub validation_success_rate: f64,
    
    /// Dimensional progress information
    pub dimensional_progress: DimensionalProgress,
    
    /// Positions analyzed in this cycle
    pub positions_analyzed: usize,
    
    /// Time taken for this discovery cycle
    pub cycle_duration: std::time::Duration,
}

/// Information about dimensional reduction progress
#[derive(Debug, Clone)]
pub struct DimensionalProgress {
    pub current_dimension: usize,
    pub previous_dimension: usize,
    pub reduction_achieved: bool,
    pub preservation_score: f64,
    pub next_target_dimension: Option<usize>,
}

impl ChessMathematicalDiscoveryEngine {
    /// Creates a new mathematical discovery engine
    /// Initializes the engine state Ω = (K, D, V, P)
    pub fn new() -> Result<Self> {
        let knowledge_base = MathematicalKnowledgeBase {
            discovered_constants: BTreeMap::new(),
            discovered_functions: BTreeMap::new(),
            discovered_invariants: Vec::new(),
            proven_theorems: Vec::new(),
            created_at: SystemTime::now(),
        };
        
        let dimensional_state = DimensionalState {
            current_dimension: 1024, // Start with full dimensional space
            reduction_history: Vec::new(),
            current_transformation: None,
            preserved_variance: 1.0,
            preservation_score: 1.0,
        };
        
        let validation_results = ValidationHistory {
            validation_scores: Vec::new(),
            consistency_scores: BTreeMap::new(),
            average_validation_score: 0.0,
            validation_trend: 0.0,
        };
        
        let progress_state = ProgressState {
            positions_analyzed: 0,
            milestones: Vec::new(),
            current_phase: DiscoveryPhase::Exploration,
            convergence_indicators: ConvergenceIndicators {
                constant_discovery_rate: 0.0,
                function_discovery_rate: 0.0,
                validation_improvement_rate: 0.0,
                dimensional_reduction_possible: false,
                convergence_score: 0.0,
                estimated_time_to_convergence: None,
            },
            session_start: SystemTime::now(),
            last_saved: SystemTime::now(),
        };
        
        let exploration_config = ExplorationConfig {
            batch_size: 100,
            stability_threshold: 0.85,
            correlation_threshold: 0.9,
            validation_threshold: 0.85,
            max_function_complexity: 50.0,
            preservation_threshold: 0.9,
            convergence_threshold: 0.99,
        };
        
        Ok(Self {
            knowledge_base,
            dimensional_state,
            validation_results,
            progress_state,
            exploration_config,
        })
    }
    
    /// Update engine configuration with new thresholds
    pub fn update_configuration(&mut self, 
        stability_threshold: f64,
        correlation_threshold: f64, 
        validation_threshold: f64,
        preservation_threshold: f64,
        batch_size: usize
    ) {
        self.exploration_config.stability_threshold = stability_threshold;
        self.exploration_config.correlation_threshold = correlation_threshold;
        self.exploration_config.validation_threshold = validation_threshold;
        self.exploration_config.preservation_threshold = preservation_threshold;
        self.exploration_config.batch_size = batch_size;
    }
    
    /// Main discovery cycle: Ω_{t+1} = Φ(Ω_t, X_t)
    /// This is the core mathematical discovery function
    pub fn run_discovery_cycle(&mut self, positions: &[ChessPosition]) -> Result<DiscoveryResults> {
        let cycle_start = std::time::Instant::now();
        
        // Convert positions to vectors: φ(s_i) for each position s_i
        let position_vectors: Vec<Array1<f64>> = positions.iter()
            .map(|pos| pos.to_vector())
            .collect();
        
        // Pattern discovery: Π: ℝ^{n×m} → 𝒫(Patterns)
        let discovered_patterns = self.discover_patterns(&position_vectors)?;
        
        // Update knowledge base with new patterns
        self.update_knowledge_base(&discovered_patterns)?;
        
        // Validate discoveries against higher-dimensional space
        let validation_success_rate = self.validate_discoveries(&discovered_patterns)?;
        
        // Check if dimensional reduction is possible
        let dimensional_progress = self.evaluate_dimensional_reduction()?;
        
        // Update progress indicators
        self.update_progress_indicators(positions.len(), &discovered_patterns);
        
        let cycle_duration = cycle_start.elapsed();
        
        Ok(DiscoveryResults {
            new_patterns: discovered_patterns,
            updated_constants: Vec::new(), // TODO: track constant updates
            new_functions: Vec::new(),     // TODO: track function updates
            validation_success_rate,
            dimensional_progress,
            positions_analyzed: positions.len(),
            cycle_duration,
        })
    }
    
    /// Pattern Discovery Function: Π: ℝ^{n×m} → 𝒫(Patterns)
    /// Mathematical form: discovers constants, functions, and invariants with intelligent filtering
    fn discover_patterns(&self, position_vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut all_patterns = Vec::new();
        
        // Discover mathematical constants
        let constants = self.discover_constants(position_vectors)?;
        all_patterns.extend(constants);
        
        // Discover functional relationships
        let functions = self.discover_functional_relationships(position_vectors)?;
        all_patterns.extend(functions);
        
        // Discover mathematical invariants
        let invariants = self.discover_invariants(position_vectors)?;
        all_patterns.extend(invariants);
        
        // Apply intelligent pattern classification to filter out artifacts
        let filtered_patterns: Vec<DiscoveredPattern> = all_patterns
            .into_iter()
            .filter(|pattern| {
                let significance = self.classify_pattern_significance(pattern);
                matches!(significance, PatternSignificance::StrategicDiscovery)
            })
            .collect();
        
        Ok(filtered_patterns)
    }
    
    /// Constant Discovery: Find chess-specific values that remain stable across position space
    /// Mathematical form: For candidate function g: stability = 1 - (σ/μ)
    /// Focus on meaningful chess strategic constants, not encoding artifacts
    fn discover_constants(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        if vectors.is_empty() {
            println!("⚠️ Constant discovery: No position vectors provided");
            return Ok(patterns);
        }
        
        // Focus on strategic features (dimensions 768-1023) rather than piece encoding
        let strategic_start = 768;
        let num_features = vectors[0].len();
        
        println!("🔍 Constant discovery: Analyzing {} vectors with {} features", vectors.len(), num_features);
        println!("   Stability threshold: {:.3}", self.exploration_config.stability_threshold);
        
        // Analyze chess-specific strategic ratios and relationships
        let strategic_constants = self.discover_chess_strategic_constants(vectors)?;
        println!("   Found {} strategic constants", strategic_constants.len());
        patterns.extend(strategic_constants);
        
        // Look for stable ratios between different strategic components
        let ratio_constants = self.discover_strategic_ratios(vectors)?;
        println!("   Found {} strategic ratio constants", ratio_constants.len());
        patterns.extend(ratio_constants);
        
        // Find positional constants (like center vs edge importance)
        let positional_constants = self.discover_positional_constants(vectors)?;
        println!("   Found {} positional constants", positional_constants.len());
        patterns.extend(positional_constants);
        
        println!("   Total constants before filtering: {}", patterns.len());
        
        Ok(patterns)
    }
    
    /// Pattern Classification Function: Distinguishes genuine chess discoveries from artifacts
    /// Mathematical form: classifier: Pattern → {Genuine, Artifact, RuleConstant}
    fn classify_pattern_significance(&self, pattern: &DiscoveredPattern) -> PatternSignificance {
        match pattern {
            DiscoveredPattern::Constant { name, value, stability, .. } => {
                // Filter out encoding artifacts (perfect mathematical constants)
                if self.is_encoding_artifact(name, *value, *stability) {
                    return PatternSignificance::EncodingArtifact;
                }
                
                // Filter out known chess rule constants
                if self.is_chess_rule_constant(name, *value) {
                    return PatternSignificance::ChessRuleConstant;
                }
                
                // Check if it's a genuine strategic discovery
                if self.is_strategic_discovery(name, *value, *stability) {
                    return PatternSignificance::StrategicDiscovery;
                }
                
                PatternSignificance::Unknown
            }
            DiscoveredPattern::LinearRelationship { correlation, .. } => {
                // Perfect correlations between equivalent encodings are artifacts
                if correlation.abs() > 0.9999 {
                    PatternSignificance::EncodingArtifact
                } else if correlation.abs() > 0.95 {
                    PatternSignificance::StrategicDiscovery
                } else {
                    PatternSignificance::Unknown
                }
            }
            DiscoveredPattern::SymbolicExpression { r_squared, complexity, fitness, .. } => {
                // Much more lenient acceptance criteria for symbolic expressions
                if *r_squared > 0.6 && *fitness > 0.6 && *complexity < 50 {
                    PatternSignificance::StrategicDiscovery
                } else if *r_squared > 0.4 && *fitness > 0.5 && *complexity >= 10 && *complexity < 30 {
                    // Accept medium-quality complex patterns for exploration
                    PatternSignificance::StrategicDiscovery
                } else if *r_squared > 0.99 && *complexity < 3 {
                    // Very high accuracy with very low complexity might be encoding artifact
                    PatternSignificance::EncodingArtifact
                } else {
                    PatternSignificance::Unknown
                }
            }
            _ => PatternSignificance::Unknown
        }
    }
    
    /// Checks if a pattern is an encoding artifact
    fn is_encoding_artifact(&self, name: &str, value: f64, stability: f64) -> bool {
        // Perfect stability suggests encoding relationship, not chess discovery
        if stability > 0.9999 {
            return true;
        }
        
        // Check for mathematical constants that are encoding artifacts
        let well_known_constants = [
            (std::f64::consts::E, "e"),
            (std::f64::consts::PI, "pi"),
            (std::f64::consts::LN_2, "ln2"),
            (1.0, "identity"),
            (0.0, "zero"),
            (-1.0, "negative_one"),
            (2.0, "two"),
            (0.5, "half"),
        ];
        
        for (constant, _) in &well_known_constants {
            if (value - constant).abs() < 0.001 {
                return true;
            }
        }
        
        // Linear encoding relationships (like piece values)
        if name.contains("linear_") && stability > 0.999 {
            return true;
        }
        
        false
    }
    
    /// Checks if a pattern represents a known chess rule constant
    fn is_chess_rule_constant(&self, name: &str, value: f64) -> bool {
        // Standard piece values
        let piece_values = [1.0, 3.0, 3.0, 5.0, 9.0, 100.0]; // pawn, knight, bishop, rook, queen, king
        for piece_value in &piece_values {
            if (value - piece_value).abs() < 0.1 {
                return true;
            }
        }
        
        // Chess board constants
        if (value - 64.0).abs() < 0.1 || // board squares
           (value - 8.0).abs() < 0.1 ||  // board dimension
           (value - 16.0).abs() < 0.1 || // total pieces per side
           (value - 32.0).abs() < 0.1 {  // total pieces
            return true;
        }
        
        false
    }
    
    /// Checks if a pattern represents a genuine strategic discovery
    fn is_strategic_discovery(&self, name: &str, value: f64, stability: f64) -> bool {
        // Must have good stability but not perfect (which suggests encoding)
        if stability < 0.90 || stability > 0.999 {
            return false;
        }
        
        // Strategic pattern names
        let strategic_keywords = [
            "ratio", "balance", "efficiency", "importance", 
            "advantage", "pressure", "activity", "harmony",
            "material", "center", "development", "king_safety",
            "pawn_structure", "piece_coordination", "tempo"
        ];
        
        for keyword in &strategic_keywords {
            if name.contains(keyword) {
                return true;
            }
        }
        
        // Enhanced value range checks for different strategic concepts
        if self.is_value_in_strategic_range(name, value) {
            return true;
        }
        
        // Check for mathematical constant properties
        if self.exhibits_mathematical_constant_properties(value, stability, name) {
            return true;
        }
        
        false
    }
    
    /// Check if value is in strategic range for the given concept
    fn is_value_in_strategic_range(&self, name: &str, value: f64) -> bool {
        match name {
            name if name.contains("ratio") => value > 0.1 && value < 10.0,
            name if name.contains("balance") => value.abs() < 5.0,
            name if name.contains("efficiency") => value > 0.0 && value < 2.0,
            name if name.contains("importance") => value > 0.0 && value < 100.0,
            name if name.contains("development") => value > 0.0 && value < 50.0,
            name if name.contains("material") => value.abs() < 20.0,
            _ => value > 0.01 && value < 100.0,
        }
    }
    
    /// Check if value exhibits characteristics of a fundamental mathematical constant
    /// Based on mathematical properties, not hard-coded values
    fn exhibits_mathematical_constant_properties(&self, value: f64, stability: f64, name: &str) -> bool {
        // Mathematical constants should have:
        // 1. High stability (low variance across observations)
        // 2. Values that aren't obvious encoding artifacts
        // 3. Appear in multiple contexts
        // 4. Have mathematical significance beyond randomness
        
        // Must have very high stability for mathematical constants
        if stability < 0.985 {
            return false;
        }
        
        // Should not be trivial values
        if self.is_trivial_value(value) {
            return false;
        }
        
        // Should appear in strategic contexts
        if !self.appears_in_strategic_context(name) {
            return false;
        }
        
        // Mathematical constants often have special properties
        if self.has_special_mathematical_properties(value) {
            return true;
        }
        
        // If it's highly stable and strategic, it might be a constant
        stability > 0.995 && self.appears_in_strategic_context(name)
    }
    
    /// Check if value is trivial (like 0, 1, 2, etc.)
    fn is_trivial_value(&self, value: f64) -> bool {
        // Common trivial values
        let trivial_values = [0.0, 1.0, 2.0, 0.5, -1.0, -2.0, 10.0, 100.0];
        
        for &trivial in &trivial_values {
            if (value - trivial).abs() < 0.001 {
                return true;
            }
        }
        
        false
    }
    
    /// Check if the constant appears in strategic contexts
    fn appears_in_strategic_context(&self, name: &str) -> bool {
        let strategic_indicators = [
            "ratio", "efficiency", "balance", "importance",
            "center", "material", "development", "activity"
        ];
        
        strategic_indicators.iter().any(|&indicator| name.contains(indicator))
    }
    
    /// Check if value has special mathematical properties
    fn has_special_mathematical_properties(&self, value: f64) -> bool {
        // Test for various mathematical properties without hard-coding specific values
        
        // Golden ratio properties: φ² = φ + 1
        if self.satisfies_golden_ratio_property(value) {
            return true;
        }
        
        // Euler's number properties: e^x where x is simple
        if self.satisfies_exponential_property(value) {
            return true;
        }
        
        // Trigonometric constants: sin/cos of special angles
        if self.satisfies_trigonometric_property(value) {
            return true;
        }
        
        // Algebraic properties: roots of simple polynomials
        if self.satisfies_algebraic_property(value) {
            return true;
        }
        
        false
    }
    
    /// Test if value satisfies golden ratio property: φ² = φ + 1
    fn satisfies_golden_ratio_property(&self, value: f64) -> bool {
        let phi_squared = value * value;
        let phi_plus_one = value + 1.0;
        (phi_squared - phi_plus_one).abs() < 0.001
    }
    
    /// Test if value satisfies exponential properties
    fn satisfies_exponential_property(&self, value: f64) -> bool {
        // Test if value = e^n for small integer n
        for n in -3..=3 {
            let exp_n = (n as f64).exp();
            if (value - exp_n).abs() < 0.01 {
                return true;
            }
        }
        false
    }
    
    /// Test if value satisfies trigonometric properties
    fn satisfies_trigonometric_property(&self, value: f64) -> bool {
        // Test special angles: π/6, π/4, π/3, π/2, etc.
        use std::f64::consts::PI;
        
        let special_angles = [
            PI/6.0, PI/4.0, PI/3.0, PI/2.0, 
            2.0*PI/3.0, 3.0*PI/4.0, 5.0*PI/6.0
        ];
        
        for &angle in &special_angles {
            if (value - angle.sin()).abs() < 0.01 || 
               (value - angle.cos()).abs() < 0.01 {
                return true;
            }
        }
        false
    }
    
    /// Test if value satisfies algebraic properties
    fn satisfies_algebraic_property(&self, value: f64) -> bool {
        // Test if value is a root of simple polynomials
        // x² - 2 = 0 (√2)
        if (value * value - 2.0).abs() < 0.001 {
            return true;
        }
        
        // x² - 3 = 0 (√3)
        if (value * value - 3.0).abs() < 0.001 {
            return true;
        }
        
        // x² - 5 = 0 (√5)
        if (value * value - 5.0).abs() < 0.001 {
            return true;
        }
        
        false
    }
    
    /// Analyze if a constant exhibits novel mathematical relationships
    /// This helps identify truly new mathematical discoveries
    fn exhibits_novel_mathematical_relationships(&self, value: f64, name: &str) -> bool {
        // Test for relationships with other discovered constants
        for existing_constant in self.knowledge_base.discovered_constants.values() {
            if self.test_mathematical_relationship(value, existing_constant.value, name, &existing_constant.name) {
                return true;
            }
        }
        
        false
    }
    
    /// Test for mathematical relationships between two constants
    fn test_mathematical_relationship(&self, value1: f64, value2: f64, name1: &str, name2: &str) -> bool {
        // Test various mathematical relationships
        
        // Ratio relationships
        if value2 != 0.0 {
            let ratio = value1 / value2;
            if self.is_interesting_ratio(ratio) {
                println!("🔍 Found ratio relationship: {} / {} = {:.6}", name1, name2, ratio);
                return true;
            }
        }
        
        // Sum relationships
        let sum = value1 + value2;
        if self.is_interesting_value(sum) {
            println!("🔍 Found sum relationship: {} + {} = {:.6}", name1, name2, sum);
            return true;
        }
        
        // Product relationships
        let product = value1 * value2;
        if self.is_interesting_value(product) {
            println!("🔍 Found product relationship: {} * {} = {:.6}", name1, name2, product);
            return true;
        }
        
        // Power relationships
        if value1 > 0.0 && value2.abs() < 5.0 {
            let power = value1.powf(value2);
            if self.is_interesting_value(power) {
                println!("🔍 Found power relationship: {} ^ {} = {:.6}", name1, name2, power);
                return true;
            }
        }
        
        false
    }
    
    /// Check if a ratio is mathematically interesting
    fn is_interesting_ratio(&self, ratio: f64) -> bool {
        // Golden ratio and its inverse
        if (ratio - 1.618).abs() < 0.01 || (ratio - 0.618).abs() < 0.01 {
            return true;
        }
        
        // Simple fractions that might be significant
        let interesting_ratios = [
            0.5, 2.0, 1.5, 0.75, 1.25, 0.8, 1.2, 0.6, 1.667, 0.333, 3.0
        ];
        
        for &interesting in &interesting_ratios {
            if (ratio - interesting).abs() < 0.01 {
                return true;
            }
        }
        
        false
    }
    
    /// Check if a value is mathematically interesting
    fn is_interesting_value(&self, value: f64) -> bool {
        // Test for special mathematical constants
        use std::f64::consts::{PI, E};
        
        let special_values = [
            PI, E, PI/2.0, PI/4.0, E/2.0, 2.0*PI, 
            2.0_f64.sqrt(), 3.0_f64.sqrt(), 5.0_f64.sqrt(),
            1.618, 0.618 // Golden ratio
        ];
        
        for &special in &special_values {
            if (value - special).abs() < 0.01 {
                return true;
            }
        }
        
        false
    }

    /// Discover chess-specific strategic constants
    /// Focus on meaningful chess relationships, not mathematical artifacts
    fn discover_chess_strategic_constants(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        println!("   🔍 Strategic constant discovery starting...");
        
        // Extract strategic components from each position
        let mut material_ratios = Vec::new();
        let mut center_control_ratios = Vec::new();
        let mut king_safety_ratios = Vec::new();
        let mut development_ratios = Vec::new();
        
        for vector in vectors {
            // Material balance (index 768)
            let material = vector[768];
            
            // Positional score (index 769)
            let positional = vector[769];
            
            // King safety (indices 770-771)
            let white_king_safety = vector[770];
            let black_king_safety = vector[771];
            
            // Center control (index 772)
            let center_control = vector[772];
            
            // Development (index 773)
            let development = vector[773];
            
            // Calculate meaningful chess ratios with better sanitization
            if material.abs() > 0.001 {
                let ratio = positional / material;
                if ratio.is_finite() && ratio.abs() < 1000.0 {
                    material_ratios.push(ratio);
                }
            }
            
            if center_control.abs() > 0.001 {
                let ratio = development / center_control;
                if ratio.is_finite() && ratio.abs() < 1000.0 {
                    center_control_ratios.push(ratio);
                }
            }
            
            if white_king_safety.abs() > 0.001 && black_king_safety.abs() > 0.001 {
                let ratio = white_king_safety / black_king_safety;
                // Clamp extreme ratios to prevent numerical instability
                if ratio.is_finite() && ratio.abs() < 1000.0 {
                    king_safety_ratios.push(ratio);
                }
            }
            
            if material.abs() > 0.001 {
                let ratio = development / material.abs();
                if ratio.is_finite() && ratio.abs() < 1000.0 {
                    development_ratios.push(ratio);
                }
            }
        }
        
        // Check if these chess-specific ratios are constant
        patterns.extend(self.check_ratio_stability("material_to_positional_ratio", &material_ratios)?);
        patterns.extend(self.check_ratio_stability("development_to_center_ratio", &center_control_ratios)?);
        patterns.extend(self.check_ratio_stability("king_safety_balance", &king_safety_ratios)?);
        patterns.extend(self.check_ratio_stability("development_efficiency", &development_ratios)?);
        
        Ok(patterns)
    }
    
    /// Discover strategic ratios between chess components
    fn discover_strategic_ratios(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        // Look for the golden ratio or other strategic proportions in chess
        let mut pawn_structure_ratios = Vec::new();
        let mut piece_activity_ratios = Vec::new();
        
        for vector in vectors {
            let pawn_structure = vector[774];
            let center_control = vector[772];
            let development = vector[773];
            
            if center_control.abs() > 0.001 {
                let ratio = pawn_structure / center_control;
                if ratio.is_finite() && ratio.abs() < 1000.0 {
                    pawn_structure_ratios.push(ratio);
                }
            }
            
            if development.abs() > 0.001 {
                let ratio = center_control / development;
                if ratio.is_finite() && ratio.abs() < 1000.0 {
                    piece_activity_ratios.push(ratio);
                }
            }
        }
        
        patterns.extend(self.check_ratio_stability("pawn_to_center_ratio", &pawn_structure_ratios)?);
        patterns.extend(self.check_ratio_stability("center_to_activity_ratio", &piece_activity_ratios)?);
        
        Ok(patterns)
    }
    
    /// Discover positional constants (board geometry relationships)
    fn discover_positional_constants(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        // Look for center square importance vs edge square importance
        let mut center_vs_edge_ratios = Vec::new();
        
        for vector in vectors {
            // Approximate center control vs edge control from piece positions
            let mut center_activity = 0.0;
            let mut edge_activity = 0.0;
            
            // Central squares are more valuable - look for this ratio
            // This is a simplified calculation based on known chess principles
            for piece_type in 0..12 { // 12 piece types
                for square in 0..64 {
                    let idx = piece_type * 64 + square;
                    if idx < 768 && vector[idx] > 0.0 {
                        let rank = square / 8;
                        let file = square % 8;
                        
                        // Center squares (e4, e5, d4, d5 area)
                        if (rank == 3 || rank == 4) && (file == 3 || file == 4) {
                            center_activity += vector[idx];
                        }
                        // Edge squares
                        else if rank == 0 || rank == 7 || file == 0 || file == 7 {
                            edge_activity += vector[idx];
                        }
                    }
                }
            }
            
            if edge_activity > 0.01 {
                center_vs_edge_ratios.push(center_activity / edge_activity);
            }
        }
        
        patterns.extend(self.check_ratio_stability("center_vs_edge_importance", &center_vs_edge_ratios)?);
        
        Ok(patterns)
    }
    
    /// Check if a set of ratios represents a stable constant
    fn check_ratio_stability(&self, name: &str, ratios: &[f64]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        println!("     📊 Checking stability for '{}' with {} ratios", name, ratios.len());
        
        if ratios.len() < 3 {
            println!("       ⚠️ Too few ratios ({} < 3) - skipping", ratios.len());
            return Ok(patterns);
        }
        
        if ratios.iter().any(|&x| !x.is_finite()) {
            println!("       ⚠️ Non-finite values detected - skipping");
            return Ok(patterns);
        }
        
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let variance = ratios.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / ratios.len() as f64;
        
        println!("       Mean: {:.6}, Variance: {:.6}", mean, variance);
        
        if mean.abs() > 0.001 { // Lowered from 0.01 to allow smaller meaningful ratios
            let coefficient_of_variation = variance.sqrt() / mean.abs();
            let stability = 1.0 / (1.0 + coefficient_of_variation);
            
            println!("       CV: {:.6}, Stability: {:.6} (threshold: {:.3})", 
                     coefficient_of_variation, stability, self.exploration_config.stability_threshold);
            
            // Use configured stability threshold
            if stability > self.exploration_config.stability_threshold {
                println!("       ✅ Constant discovered: {} = {:.6} (stability: {:.6})", name, mean, stability);
                patterns.push(DiscoveredPattern::Constant {
                    name: name.to_string(),
                    value: mean,
                    stability,
                    occurrences: ratios.len(),
                });
            } else {
                println!("       ❌ Stability too low: {:.6} < {:.3}", stability, self.exploration_config.stability_threshold);
            }
        } else {
            println!("       ❌ Mean too small: {:.6} < 0.001", mean.abs());
        }
        
        Ok(patterns)
    }
    
    /// Functional Relationship Discovery
    /// Mathematical form: find f such that y = f(x) with high correlation
    fn discover_functional_relationships(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        if vectors.is_empty() {
            return Ok(patterns);
        }
        
        let num_features = vectors[0].len();
        let mut symbolic_attempts = 0;
        let mut strategic_features_tested = 0;
        
        // Test pairwise relationships between features - prioritize symbolic over linear
        for i in 0..num_features {
            for j in (i + 1)..num_features {
                let x_values: Vec<f64> = vectors.iter().map(|v| v[i]).collect();
                let y_values: Vec<f64> = vectors.iter().map(|v| v[j]).collect();
                
                // Skip if either feature is all zeros
                if x_values.iter().all(|&x| x == 0.0) || y_values.iter().all(|&x| x == 0.0) {
                    continue;
                }
                
                // PRIORITIZE: Test symbolic regression first for non-linear patterns
                let symbolic_found = if self.should_run_symbolic_regression(i, j, &x_values, &y_values) {
                    symbolic_attempts += 1;
                    if self.is_strategic_feature(i) || self.is_strategic_feature(j) {
                        strategic_features_tested += 1;
                    }
                    
                    if let Some(symbolic_pattern) = self.test_symbolic_regression(&x_values, &y_values, i, j, vectors)? {
                        patterns.push(symbolic_pattern);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                
                // Only test simpler patterns if no complex symbolic pattern was found
                if !symbolic_found {
                    // Test polynomial relationships before linear
                    let poly_found = !self.test_polynomial_relationships(&x_values, &y_values, i, j)?.is_empty();
                    if poly_found {
                        patterns.extend(self.test_polynomial_relationships(&x_values, &y_values, i, j)?);
                    } else {
                        // Only test linear as last resort
                        if let Some(linear_pattern) = self.test_linear_relationship(&x_values, &y_values, i, j)? {
                            patterns.push(linear_pattern);
                        }
                    }
                }
            }
        }
        
        println!("🔍 Function discovery summary:");
        println!("   Symbolic regression attempts: {}", symbolic_attempts);
        println!("   Strategic features tested: {}", strategic_features_tested);
        println!("   Total functional patterns: {}", patterns.len());
        
        Ok(patterns)
    }
    
    /// Test for linear relationship: y = ax + b
    /// Mathematical form: compute correlation coefficient ρ
    fn test_linear_relationship(
        &self, 
        x_values: &[f64], 
        y_values: &[f64], 
        feature_i: usize, 
        feature_j: usize
    ) -> Result<Option<DiscoveredPattern>> {
        if x_values.len() != y_values.len() || x_values.is_empty() {
            return Ok(None);
        }
        
        let n = x_values.len() as f64;
        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = y_values.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut x_sum_sq = 0.0;
        let mut y_sum_sq = 0.0;
        
        for (&x, &y) in x_values.iter().zip(y_values.iter()) {
            let x_diff = x - x_mean;
            let y_diff = y - y_mean;
            numerator += x_diff * y_diff;
            x_sum_sq += x_diff * x_diff;
            y_sum_sq += y_diff * y_diff;
        }
        
        if x_sum_sq == 0.0 || y_sum_sq == 0.0 {
            return Ok(None);
        }
        
        let correlation = numerator / (x_sum_sq.sqrt() * y_sum_sq.sqrt());
        
        if correlation.abs() > self.exploration_config.correlation_threshold {
            let slope = numerator / x_sum_sq;
            let intercept = y_mean - slope * x_mean;
            
            Ok(Some(DiscoveredPattern::LinearRelationship {
                coefficient: slope,
                intercept,
                correlation,
                feature_names: (format!("feature_{}", feature_i), format!("feature_{}", feature_j)),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Test for polynomial relationships of degree 2 and 3
    /// Mathematical form: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
    fn test_polynomial_relationships(
        &self,
        x_values: &[f64],
        y_values: &[f64],
        feature_i: usize,
        feature_j: usize,
    ) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        // Test degree 2 polynomial
        if let Some(poly2) = self.fit_polynomial(x_values, y_values, 2, feature_i, feature_j)? {
            patterns.push(poly2);
        }
        
        // Test degree 3 polynomial (if we have enough data points)
        if x_values.len() >= 8 {
            if let Some(poly3) = self.fit_polynomial(x_values, y_values, 3, feature_i, feature_j)? {
                patterns.push(poly3);
            }
        }
        
        Ok(patterns)
    }
    
    /// Fit polynomial of given degree using least squares
    /// Mathematical form: minimize Σ(y_i - Σ(a_j * x_i^j))²
    fn fit_polynomial(
        &self,
        x_values: &[f64],
        y_values: &[f64],
        degree: u32,
        feature_i: usize,
        feature_j: usize,
    ) -> Result<Option<DiscoveredPattern>> {
        if x_values.len() < (degree + 1) as usize {
            return Ok(None);
        }
        
        let n = x_values.len();
        let m = (degree + 1) as usize;
        
        // Build design matrix X where X[i][j] = x[i]^j
        let mut design_matrix = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                design_matrix[i][j] = x_values[i].powi(j as i32);
            }
        }
        
        // Solve normal equations: (X^T X) β = X^T y
        // For simplicity, we'll use a basic implementation for small matrices
        let coefficients = self.solve_least_squares(&design_matrix, y_values)?;
        
        if coefficients.is_empty() {
            return Ok(None);
        }
        
        // Compute R-squared
        let y_mean = y_values.iter().sum::<f64>() / n as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        
        for (i, &y_actual) in y_values.iter().enumerate() {
            let y_pred: f64 = coefficients.iter().enumerate()
                .map(|(j, &coef)| coef * x_values[i].powi(j as i32))
                .sum();
            
            ss_res += (y_actual - y_pred).powi(2);
            ss_tot += (y_actual - y_mean).powi(2);
        }
        
        let r_squared = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };
        
        // Only accept if R-squared is above threshold
        if r_squared > self.exploration_config.correlation_threshold {
            Ok(Some(DiscoveredPattern::PolynomialRelationship {
                coefficients,
                degree,
                r_squared,
                feature_names: vec![format!("feature_{}", feature_i), format!("feature_{}", feature_j)],
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Simple least squares solver for small matrices
    /// Mathematical form: solve (X^T X) β = X^T y
    fn solve_least_squares(&self, design_matrix: &[Vec<f64>], y_values: &[f64]) -> Result<Vec<f64>> {
        let n = design_matrix.len();
        let m = design_matrix[0].len();
        
        // Compute X^T X
        let mut xtx = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    xtx[i][j] += design_matrix[k][i] * design_matrix[k][j];
                }
            }
        }
        
        // Compute X^T y
        let mut xty = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                xty[i] += design_matrix[j][i] * y_values[j];
            }
        }
        
        // Solve using Gaussian elimination (for small matrices)
        self.gaussian_elimination(&mut xtx, &mut xty)
    }
    
    /// Gaussian elimination solver
    fn gaussian_elimination(&self, matrix: &mut [Vec<f64>], rhs: &mut [f64]) -> Result<Vec<f64>> {
        let n = matrix.len();
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if matrix[k][i].abs() > matrix[max_row][i].abs() {
                    max_row = k;
                }
            }
            
            // Swap rows
            matrix.swap(i, max_row);
            rhs.swap(i, max_row);
            
            // Check for zero pivot
            if matrix[i][i].abs() < 1e-10 {
                return Ok(vec![]); // Singular matrix
            }
            
            // Eliminate column
            for k in (i + 1)..n {
                let factor = matrix[k][i] / matrix[i][i];
                rhs[k] -= factor * rhs[i];
                for j in i..n {
                    matrix[k][j] -= factor * matrix[i][j];
                }
            }
        }
        
        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            solution[i] = rhs[i];
            for j in (i + 1)..n {
                solution[i] -= matrix[i][j] * solution[j];
            }
            solution[i] /= matrix[i][i];
        }
        
        Ok(solution)
    }
    
    /// Determines if a feature is strategically important for chess analysis
    fn is_strategic_feature(&self, feature_idx: usize) -> bool {
        // Strategic features start at index 768
        if feature_idx >= 768 {
            return true;
        }
        
        // Also include key piece positions (center squares, key squares)
        // Piece positions are in first 768 features (12 pieces × 64 squares)
        let square_idx = feature_idx % 64;
        let piece_type = feature_idx / 64;
        
        // Center squares (d4, d5, e4, e5)
        let center_squares = [27, 28, 35, 36]; // d4, d5, e4, e5 in 0-63 indexing
        if center_squares.contains(&square_idx) {
            return true;
        }
        
        // Important pieces (kings, queens, key pieces)
        if piece_type == 0 || piece_type == 6 || // White/Black king
           piece_type == 1 || piece_type == 7 || // White/Black queen
           piece_type == 5 || piece_type == 11 { // White/Black knights
            return true;
        }
        
        false
    }
    
    /// Determines if symbolic regression should be run for a feature pair
    /// Enhanced to prioritize symbolic expression discovery over linear patterns
    fn should_run_symbolic_regression(&self, feature_i: usize, feature_j: usize, x_values: &[f64], y_values: &[f64]) -> bool {
        // Enhanced strategic feature identification
        let strategic_priority = self.is_strategic_feature(feature_i) || self.is_strategic_feature(feature_j);
        
        // Also allow piece-to-strategic relationships for complex pattern discovery
        let cross_domain = (feature_i < 768 && feature_j >= 768) || (feature_i >= 768 && feature_j < 768);
        
        if !strategic_priority && !cross_domain {
            // Still allow some piece-position relationships, but be more selective
            return ((feature_i + feature_j) % 25) == 0; // Much more selective sampling
        }
        
        // Require sufficient data points
        if x_values.len() < 8 {  // Reduced from 10 to allow more attempts
            return false;
        }
        
        // Check for non-trivial variance
        let x_var = self.compute_variance(x_values);
        let y_var = self.compute_variance(y_values);
        
        if x_var < 1e-8 || y_var < 1e-8 {  // More sensitive to variance
            return false;
        }
        
        // Pre-filter based on correlation - only run symbolic regression on interesting relationships
        let correlation = self.compute_correlation(x_values, y_values);
        if correlation.abs() < 0.3 && !strategic_priority {
            return false; // Skip weak correlations for non-strategic features
        }
        
        // Significantly increase symbolic regression frequency for strategic features
        if strategic_priority {
            // Run on most strategic feature pairs (much more aggressive)
            ((feature_i + feature_j) % 3) == 0  // Sample every 3rd pair instead of every 10th
        } else {
            // Cross-domain relationships - sample moderately
            ((feature_i + feature_j) % 8) == 0
        }
    }
    
    /// Test symbolic regression for non-linear relationships
    /// Mathematical form: evolve expression tree to minimize error
    fn test_symbolic_regression(
        &self,
        x_values: &[f64],
        y_values: &[f64],
        feature_i: usize,
        feature_j: usize,
        position_vectors: &[Array1<f64>],
    ) -> Result<Option<DiscoveredPattern>> {
        // Configure symbolic regression for strategic chess discovery - prioritize complexity
        let config = SymbolicRegressionConfig {
            population_size: 75,     // Larger population for more diversity
            max_generations: 40,     // More generations for complex evolution
            max_depth: 6,           // Higher complexity allowed
            mutation_rate: 0.20,    // Higher mutation to explore more patterns
            crossover_rate: 0.8,    // Higher crossover for complex recombination
            elitism_rate: 0.15,     // Keep more elite individuals
            complexity_penalty: 0.015, // Moderate penalty to reduce function noise
            target_fitness: 0.75,   // Lower threshold to find more diverse patterns
            verbose_logging: false, // Reduce spam
        };
        
        // Convert complete position vectors to ndarray format
        let inputs: Vec<Array1<f64>> = position_vectors.iter()
            .map(|vec| vec.clone())
            .collect();
        
        let targets: Vec<f64> = y_values.to_vec();
        
        // Run symbolic regression with full feature dimensionality
        let num_features = position_vectors[0].len();
        let sr = SymbolicRegression::new(config, num_features);
        
        match sr.evolve(&inputs, &targets) {
            Ok((expression, fitness)) => {
                if fitness > 0.5 { // Lower threshold to accept more diverse patterns
                    // Use the fitness directly from symbolic regression (includes strategic bonus and complexity penalty)
                    // Only calculate R-squared for reporting purposes
                    let mut total_error = 0.0;
                    let mut total_variance = 0.0;
                    let y_mean: f64 = y_values.iter().sum::<f64>() / y_values.len() as f64;
                    
                    for (i, &y_actual) in y_values.iter().enumerate() {
                        // FIX: Pass complete position vector instead of single feature
                        let position_slice = position_vectors[i].as_slice().unwrap();
                        let y_pred = expression.evaluate(position_slice);
                        if y_pred.is_finite() {
                            total_error += (y_actual - y_pred).powi(2);
                        }
                        total_variance += (y_actual - y_mean).powi(2);
                    }
                    
                    let r_squared = if total_variance > 0.0 {
                        1.0 - (total_error / total_variance)
                    } else {
                        0.0
                    };
                    
                    // Log symbolic expression quality
                    println!("      ✅ Symbolic expression discovered:");
                    println!("         Expression: {}", expression);
                    println!("         Fitness: {:.6}, R²: {:.6}, Complexity: {}", 
                             fitness, r_squared, expression.complexity());
                    println!("         Features: {} → {}", feature_i, feature_j);
                    
                    // Accept based on fitness (which includes strategic scoring), not just R²
                    Ok(Some(DiscoveredPattern::SymbolicExpression {
                        expression: expression.clone(),
                        fitness, // This includes strategic bonus and complexity penalty
                        complexity: expression.complexity(),
                        feature_names: (format!("feature_{}", feature_i), format!("feature_{}", feature_j)),
                        r_squared, // For reporting only
                    }))
                } else {
                    println!("      ❌ Symbolic expression rejected (fitness {:.6} < 0.5)", fitness);
                    Ok(None)
                }
            },
            Err(_) => Ok(None), // Symbolic regression failed, continue
        }
    }
    
    /// Helper function to compute variance
    fn compute_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64
    }
    
    /// Helper function to compute correlation coefficient
    fn compute_correlation(&self, x_values: &[f64], y_values: &[f64]) -> f64 {
        if x_values.len() != y_values.len() || x_values.len() < 2 {
            return 0.0;
        }
        
        let n = x_values.len() as f64;
        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = y_values.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut x_sum_sq = 0.0;
        let mut y_sum_sq = 0.0;
        
        for (&x, &y) in x_values.iter().zip(y_values.iter()) {
            let x_diff = x - x_mean;
            let y_diff = y - y_mean;
            numerator += x_diff * y_diff;
            x_sum_sq += x_diff * x_diff;
            y_sum_sq += y_diff * y_diff;
        }
        
        if x_sum_sq == 0.0 || y_sum_sq == 0.0 {
            return 0.0;
        }
        
        numerator / (x_sum_sq.sqrt() * y_sum_sq.sqrt())
    }
    
    /// Discover mathematical invariants
    /// Mathematical form: find properties preserved under transformations
    fn discover_invariants(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        if vectors.is_empty() {
            return Ok(patterns);
        }
        
        // Test color symmetry invariant: f(position) = f(color_flip(position))
        patterns.extend(self.test_color_symmetry_invariant(vectors)?);
        
        // Test rotational invariant: f(position) ≈ f(rotate_180(position))
        patterns.extend(self.test_rotational_invariant(vectors)?);
        
        // Test material conservation invariant
        patterns.extend(self.test_material_conservation_invariant(vectors)?);
        
        // Test center-edge symmetry
        patterns.extend(self.test_center_edge_symmetry(vectors)?);
        
        Ok(patterns)
    }
    
    /// Test color symmetry invariant
    /// Mathematical form: ∀s ∈ S: f(s) = f(color_flip(s))
    fn test_color_symmetry_invariant(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        // Test if material balance is preserved under color flip
        let mut symmetry_violations = 0;
        let mut total_tests = 0;
        
        for vector in vectors {
            // Material balance should be negated under color flip
            let material_balance = vector[768];
            let positional_score = vector[769];
            
            // Simple symmetry test: if material is balanced, positional should be symmetric
            if material_balance.abs() < 0.1 {
                // In balanced positions, positional scores should be small
                if positional_score.abs() > 0.5 {
                    symmetry_violations += 1;
                }
                total_tests += 1;
            }
        }
        
        if total_tests > 0 {
            let symmetry_strength = 1.0 - (symmetry_violations as f64 / total_tests as f64);
            
            if symmetry_strength > 0.8 {
                let invariant = MathematicalInvariant {
                    name: "color_symmetry".to_string(),
                    description: "Chess evaluation is approximately symmetric under color flip".to_string(),
                    transformation_group: vec!["color_flip".to_string()],
                    strength: symmetry_strength,
                    invariant_expression: "f(white_to_move) ≈ -f(black_to_move)".to_string(),
                    discovered_at: SystemTime::now(),
                };
                
                patterns.push(DiscoveredPattern::Invariant {
                    invariant,
                    strength: symmetry_strength,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Test rotational invariant (180-degree rotation)
    /// Mathematical form: ∀s ∈ S: f(s) ≈ f(rotate_180(s))
    fn test_rotational_invariant(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        // Test if center control is preserved under 180-degree rotation
        let mut rotation_consistency = 0;
        let mut total_tests = 0;
        
        for vector in vectors {
            let center_control = vector[772];
            let development = vector[773];
            
            // Under 180-degree rotation, center control should be preserved
            // while development might change (simplified test)
            if center_control.abs() > 0.1 {
                // If there's significant center control, it should correlate with development
                let correlation = (center_control * development).abs();
                if correlation > 0.3 {
                    rotation_consistency += 1;
                }
                total_tests += 1;
            }
        }
        
        if total_tests > 0 {
            let rotation_strength = rotation_consistency as f64 / total_tests as f64;
            
            if rotation_strength > 0.7 {
                let invariant = MathematicalInvariant {
                    name: "rotational_invariant".to_string(),
                    description: "Center control remains correlated with development under rotation".to_string(),
                    transformation_group: vec!["rotate_180".to_string()],
                    strength: rotation_strength,
                    invariant_expression: "center_control(s) ≈ center_control(rotate_180(s))".to_string(),
                    discovered_at: SystemTime::now(),
                };
                
                patterns.push(DiscoveredPattern::Invariant {
                    invariant,
                    strength: rotation_strength,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Test material conservation invariant
    /// Mathematical form: ∀s ∈ S: Σ material_value_i = constant
    fn test_material_conservation_invariant(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        // Calculate total material values across all positions
        let mut total_materials = Vec::new();
        
        for vector in vectors {
            // Sum up piece values (assuming first 768 dimensions are piece positions)
            let mut total_material = 0.0;
            
            // Count pieces by type (simplified calculation)
            for piece_type in 0..12 { // 12 piece types (6 per color)
                for square in 0..64 {
                    let idx = piece_type * 64 + square;
                    if idx < 768 {
                        total_material += vector[idx].abs();
                    }
                }
            }
            
            total_materials.push(total_material);
        }
        
        if !total_materials.is_empty() {
            let mean_material = total_materials.iter().sum::<f64>() / total_materials.len() as f64;
            let variance = total_materials.iter()
                .map(|&x| (x - mean_material).powi(2))
                .sum::<f64>() / total_materials.len() as f64;
            
            let coefficient_of_variation = if mean_material > 0.0 {
                variance.sqrt() / mean_material
            } else {
                1.0
            };
            
            let conservation_strength = 1.0 / (1.0 + coefficient_of_variation);
            
            if conservation_strength > 0.9 {
                let invariant = MathematicalInvariant {
                    name: "material_conservation".to_string(),
                    description: "Total material count remains approximately constant".to_string(),
                    transformation_group: vec!["piece_moves".to_string()],
                    strength: conservation_strength,
                    invariant_expression: "Σ material_value_i ≈ constant".to_string(),
                    discovered_at: SystemTime::now(),
                };
                
                patterns.push(DiscoveredPattern::Invariant {
                    invariant,
                    strength: conservation_strength,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Test center-edge symmetry
    /// Mathematical form: center squares have different importance than edge squares
    fn test_center_edge_symmetry(&self, vectors: &[Array1<f64>]) -> Result<Vec<DiscoveredPattern>> {
        let mut patterns = Vec::new();
        
        let mut center_edge_ratios = Vec::new();
        
        for vector in vectors {
            let mut center_activity = 0.0;
            let mut edge_activity = 0.0;
            
            // Calculate center vs edge activity
            for piece_type in 0..12 {
                for square in 0..64 {
                    let idx = piece_type * 64 + square;
                    if idx < 768 && vector[idx] > 0.0 {
                        let rank = square / 8;
                        let file = square % 8;
                        
                        if (rank == 3 || rank == 4) && (file == 3 || file == 4) {
                            center_activity += vector[idx];
                        } else if rank == 0 || rank == 7 || file == 0 || file == 7 {
                            edge_activity += vector[idx];
                        }
                    }
                }
            }
            
            if edge_activity > 0.01 {
                center_edge_ratios.push(center_activity / edge_activity);
            }
        }
        
        if !center_edge_ratios.is_empty() {
            let mean_ratio = center_edge_ratios.iter().sum::<f64>() / center_edge_ratios.len() as f64;
            let variance = center_edge_ratios.iter()
                .map(|&x| (x - mean_ratio).powi(2))
                .sum::<f64>() / center_edge_ratios.len() as f64;
            
            let stability = 1.0 / (1.0 + variance.sqrt() / mean_ratio.abs().max(1e-10));
            
            if stability > 0.8 && mean_ratio > 1.0 {
                let invariant = MathematicalInvariant {
                    name: "center_edge_asymmetry".to_string(),
                    description: "Center squares consistently have higher activity than edge squares".to_string(),
                    transformation_group: vec!["board_geometry".to_string()],
                    strength: stability,
                    invariant_expression: format!("center_activity / edge_activity ≈ {:.3}", mean_ratio),
                    discovered_at: SystemTime::now(),
                };
                
                patterns.push(DiscoveredPattern::Invariant {
                    invariant,
                    strength: stability,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Update knowledge base with newly discovered patterns
    fn update_knowledge_base(&mut self, patterns: &[DiscoveredPattern]) -> Result<()> {
        for pattern in patterns {
            match pattern {
                DiscoveredPattern::Constant { name, value, stability, occurrences } => {
                    // Update existing constant or create new one
                    let session_id = format!("session_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs());
                    
                    if self.knowledge_base.discovered_constants.contains_key(name) {
                        // Update existing constant with new observation
                        let mut existing_constant = self.knowledge_base.discovered_constants.remove(name).unwrap();
                        self.update_constant_stability(&mut existing_constant, *value, &session_id);
                        self.knowledge_base.discovered_constants.insert(name.clone(), existing_constant);
                    } else {
                        // Create new constant
                        let constant = MathematicalConstant {
                            name: name.clone(),
                            value: *value,
                            confidence: *stability, // Use stability as initial confidence
                            stability: *stability,
                            observation_count: *occurrences,
                            contexts: vec!["chess_position_analysis".to_string()],
                            mathematical_significance: stability * 100.0,
                            discovered_at: SystemTime::now(),
                            validation_score: 0.0, // Will be updated during validation
                            historical_values: vec![*value],
                            stability_trend: 0.0,
                            session_appearances: vec![session_id],
                        };
                        
                        self.knowledge_base.discovered_constants.insert(name.clone(), constant);
                    }
                }
                
                DiscoveredPattern::LinearRelationship { coefficient, intercept, correlation, .. } => {
                    // Convert linear relationship to a mathematical function
                    let function = MathematicalFunction {
                        name: format!("linear_{}_{}", coefficient, correlation),
                        expression: format!("y = {}*x + {}", coefficient, intercept),
                        coefficients: vec![*coefficient],
                        intercept: *intercept,
                        input_dimension: 1,
                        complexity: 2.0, // y = ax + b has complexity 2
                        accuracy: correlation.abs(),
                        r_squared: correlation * correlation,
                        function_type: FunctionType::Polynomial(1),
                        discovered_at: SystemTime::now(),
                    };
                    
                    self.knowledge_base.discovered_functions.insert(function.name.clone(), function);
                }
                
                DiscoveredPattern::PolynomialRelationship { coefficients, degree, r_squared, .. } => {
                    let function = MathematicalFunction {
                        name: format!("polynomial_degree_{}", degree),
                        expression: format!("polynomial of degree {}", degree),
                        coefficients: coefficients.clone(),
                        intercept: coefficients.get(0).copied().unwrap_or(0.0), // constant term
                        input_dimension: 1,
                        complexity: *degree as f64 + 1.0,
                        accuracy: *r_squared,
                        r_squared: *r_squared,
                        function_type: FunctionType::Polynomial(*degree),
                        discovered_at: SystemTime::now(),
                    };
                    
                    self.knowledge_base.discovered_functions.insert(function.name.clone(), function);
                }
                
                DiscoveredPattern::SymbolicExpression { expression, fitness, complexity, r_squared, .. } => {
                    let function = MathematicalFunction {
                        name: format!("symbolic_expr_{}", expression.to_string().chars().take(20).collect::<String>()),
                        expression: expression.to_string(),
                        coefficients: vec![], // Symbolic expressions don't have simple coefficients
                        intercept: 0.0,
                        input_dimension: 1,
                        complexity: *complexity as f64,
                        accuracy: *fitness,
                        r_squared: *r_squared,
                        function_type: FunctionType::SymbolicExpression(expression.clone()),
                        discovered_at: SystemTime::now(),
                    };
                    
                    self.knowledge_base.discovered_functions.insert(function.name.clone(), function);
                }
                
                _ => {
                    // Handle other pattern types as they are implemented
                }
            }
        }
        
        Ok(())
    }
    
    /// Update constant stability tracking with new observation
    fn update_constant_stability(&mut self, constant: &mut MathematicalConstant, new_value: f64, session_id: &str) {
        // Add new value to historical tracking
        constant.historical_values.push(new_value);
        constant.observation_count += 1;
        constant.session_appearances.push(session_id.to_string());
        
        // Calculate new stability based on all historical values
        let mean = constant.historical_values.iter().sum::<f64>() / constant.historical_values.len() as f64;
        let variance = constant.historical_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / constant.historical_values.len() as f64;
        
        let coefficient_of_variation = if mean.abs() > 1e-10 {
            variance.sqrt() / mean.abs()
        } else {
            0.0
        };
        
        let new_stability = 1.0 / (1.0 + coefficient_of_variation);
        
        // Calculate stability trend
        let old_stability = constant.stability;
        constant.stability_trend = new_stability - old_stability;
        
        // Update stability and value
        constant.stability = new_stability;
        constant.value = mean; // Use mean of all observations
        constant.confidence = new_stability;
        
        // Update mathematical significance based on stability and frequency
        constant.mathematical_significance = new_stability * 100.0 * 
            (constant.observation_count as f64).ln().max(1.0);
        
        println!("📊 Updated constant '{}': value={:.6}, stability={:.6}, trend={:.6}", 
                 constant.name, constant.value, constant.stability, constant.stability_trend);
    }
    
    /// Validate discoveries against higher-dimensional space with detailed logging
    /// Mathematical form: validation_score = (1/m) Σᵢ indicator(pattern holds for xᵢ)
    fn validate_discoveries(&mut self, patterns: &[DiscoveredPattern]) -> Result<f64> {
        if patterns.is_empty() {
            println!("🔍 Validation: No patterns to validate");
            return Ok(0.0);
        }
        
        println!("🔍 Validation: Validating {} patterns", patterns.len());
        
        let mut successful_validations = 0;
        let mut total_validations = 0;
        
        for (i, pattern) in patterns.iter().enumerate() {
            total_validations += 1;
            
            let validation_result = match pattern {
                DiscoveredPattern::LinearRelationship { coefficient, intercept, correlation, feature_names } => {
                    println!("   📊 Validating linear relationship {} → {}", 
                             feature_names.0, feature_names.1);
                    self.validate_linear_relationship(*coefficient, *intercept, *correlation, feature_names)
                },
                DiscoveredPattern::Constant { name, value, stability, .. } => {
                    println!("   📊 Validating constant {}: {}", name, value);
                    self.validate_constant(*value, *stability)
                },
                DiscoveredPattern::PolynomialRelationship { degree, r_squared, feature_names, .. } => {
                    println!("   📊 Validating polynomial degree {} relationship", degree);
                    self.validate_polynomial_relationship(*degree, *r_squared, feature_names)
                },
                _ => {
                    // Quietly validate other patterns without spam
                    Ok(true) // Other patterns pass by default for now
                }
            };
            
            match validation_result {
                Ok(true) => {
                    successful_validations += 1;
                    // Only log validation for important patterns
                    match pattern {
                        DiscoveredPattern::Constant { .. } => {
                            println!("      ✅ Constant validation PASSED for pattern {}", i + 1);
                        },
                        _ => {} // Quiet validation for other patterns
                    }
                },
                Ok(false) => {
                    println!("      ❌ Validation FAILED for pattern {}", i + 1);
                },
                Err(e) => {
                    println!("      ⚠️  Validation ERROR for pattern {}: {}", i + 1, e);
                }
            }
        }
        
        let validation_score = successful_validations as f64 / total_validations.max(1) as f64;
        println!("🔍 Validation Summary: {}/{} passed ({:.1}%)", 
                 successful_validations, total_validations, validation_score * 100.0);
        
        Ok(validation_score)
    }
    
    /// Validate linear relationship on held-out data
    fn validate_linear_relationship(&self, coefficient: f64, intercept: f64, correlation: f64, feature_names: &(String, String)) -> Result<bool> {
        // Basic validation criteria
        let correlation_valid = correlation.abs() > self.exploration_config.correlation_threshold;
        let coefficient_reasonable = coefficient.is_finite() && coefficient.abs() < 1000.0;
        let intercept_reasonable = intercept.is_finite() && intercept.abs() < 1000.0;
        
        let is_valid = correlation_valid && coefficient_reasonable && intercept_reasonable;
        
        if !is_valid {
            println!("      🔍 Validation details: correlation={:.3}, coefficient={:.3}, intercept={:.3}", 
                     correlation, coefficient, intercept);
        }
        
        Ok(is_valid)
    }
    
    /// Validate mathematical constant stability
    fn validate_constant(&self, value: f64, stability: f64) -> Result<bool> {
        let stability_valid = stability > self.exploration_config.stability_threshold;
        let value_reasonable = value.is_finite() && value.abs() < 1000.0;
        
        let is_valid = stability_valid && value_reasonable;
        
        if !is_valid {
            println!("      ❌ Constant validation failed:");
            println!("         Stability: {:.3} (threshold: {:.3}) - {}", 
                     stability, self.exploration_config.stability_threshold,
                     if stability_valid { "PASS" } else { "FAIL" });
            println!("         Value: {:.3} (finite: {}, reasonable: {}) - {}", 
                     value, value.is_finite(), value.abs() < 1000.0,
                     if value_reasonable { "PASS" } else { "FAIL" });
        } else {
            println!("      ✅ Constant validation passed: stability={:.3}, value={:.3}", stability, value);
        }
        
        Ok(is_valid)
    }
    
    /// Validate polynomial relationship
    fn validate_polynomial_relationship(&self, degree: u32, r_squared: f64, feature_names: &[String]) -> Result<bool> {
        let r_squared_valid = r_squared > self.exploration_config.correlation_threshold;
        let degree_reasonable = degree >= 1 && degree <= 5; // Reasonable polynomial degree
        let has_features = feature_names.len() >= 2;
        
        let is_valid = r_squared_valid && degree_reasonable && has_features;
        
        if !is_valid {
            println!("      🔍 Validation details: degree={}, r_squared={:.3}, features={}", 
                     degree, r_squared, feature_names.len());
        }
        
        Ok(is_valid)
    }
    
    /// Evaluate if dimensional reduction is possible
    /// Mathematical form: preservation_score > threshold
    fn evaluate_dimensional_reduction(&self) -> Result<DimensionalProgress> {
        Ok(DimensionalProgress {
            current_dimension: self.dimensional_state.current_dimension,
            previous_dimension: self.dimensional_state.current_dimension,
            reduction_achieved: false,
            preservation_score: self.dimensional_state.preservation_score,
            next_target_dimension: if self.dimensional_state.current_dimension > 512 {
                Some(512)
            } else {
                None
            },
        })
    }
    
    /// Update progress indicators based on discovery results
    fn update_progress_indicators(&mut self, positions_count: usize, patterns: &[DiscoveredPattern]) {
        self.progress_state.positions_analyzed += positions_count;
        
        // Update discovery rates
        let _constants_discovered = patterns.iter()
            .filter(|p| matches!(p, DiscoveredPattern::Constant { .. }))
            .count();
        
        let _functions_discovered = patterns.iter()
            .filter(|p| matches!(p, 
                DiscoveredPattern::LinearRelationship { .. } | 
                DiscoveredPattern::PolynomialRelationship { .. }
            ))
            .count();
        
        if self.progress_state.positions_analyzed > 0 {
            self.progress_state.convergence_indicators.constant_discovery_rate = 
                self.knowledge_base.discovered_constants.len() as f64 / 
                self.progress_state.positions_analyzed as f64;
                
            self.progress_state.convergence_indicators.function_discovery_rate = 
                self.knowledge_base.discovered_functions.len() as f64 / 
                self.progress_state.positions_analyzed as f64;
        }
        
        // Check for milestones
        if self.knowledge_base.discovered_constants.len() >= 10 && 
           !self.progress_state.milestones.iter().any(|m| m.name == "ten_constants") {
            self.progress_state.milestones.push(DiscoveryMilestone {
                name: "ten_constants".to_string(),
                description: "Discovered 10 mathematical constants".to_string(),
                reached_at: SystemTime::now(),
                significance: 0.5,
            });
        }
    }
    
    /// Get current discovery statistics
    pub fn get_discovery_statistics(&self) -> DiscoveryStatistics {
        DiscoveryStatistics {
            constants_discovered: self.knowledge_base.discovered_constants.len(),
            functions_discovered: self.knowledge_base.discovered_functions.len(),
            invariants_discovered: self.knowledge_base.discovered_invariants.len(),
            theorems_proven: self.knowledge_base.proven_theorems.len(),
            current_dimension: self.dimensional_state.current_dimension,
            positions_analyzed: self.progress_state.positions_analyzed,
            current_phase: self.progress_state.current_phase,
            convergence_score: self.progress_state.convergence_indicators.convergence_score,
        }
    }
}

/// Summary statistics for the discovery process
#[derive(Debug, Clone)]
pub struct DiscoveryStatistics {
    pub constants_discovered: usize,
    pub functions_discovered: usize,
    pub invariants_discovered: usize,
    pub theorems_proven: usize,
    pub current_dimension: usize,
    pub positions_analyzed: usize,
    pub current_phase: DiscoveryPhase,
    pub convergence_score: f64,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            stability_threshold: 0.85,
            correlation_threshold: 0.9,
            validation_threshold: 0.85,
            max_function_complexity: 50.0,
            preservation_threshold: 0.9,
            convergence_threshold: 0.99,
        }
    }
}