/*
Symbolic Regression for Chess Mathematical Discovery

This module implements genetic programming for discovering mathematical functions
that model chess position evaluation.

Mathematical Framework:
- Expression Tree: f(x) = tree representation of mathematical function
- Fitness Function: L(f) = MSE(f(X), y) + λ * complexity(f)
- Evolution: Population evolves through selection, crossover, and mutation
*/

use ndarray::Array1;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::fmt;

/// Mathematical expression tree for symbolic regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Variable reference (feature index)
    Variable(usize),
    
    /// Constant value
    Constant(f64),
    
    /// Binary operations
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
    Divide(Box<Expression>, Box<Expression>),
    
    /// Unary operations
    Sin(Box<Expression>),
    Cos(Box<Expression>),
    Log(Box<Expression>),
    Exp(Box<Expression>),
    Sqrt(Box<Expression>),
    Abs(Box<Expression>),
    Tanh(Box<Expression>),
    Sigmoid(Box<Expression>),
    Sign(Box<Expression>),
    
    /// Power operation
    Power(Box<Expression>, Box<Expression>),
    
    /// Modulo operation for chess patterns
    Modulo(Box<Expression>, Box<Expression>),
    
    /// Min/Max operations for chess logic
    Min(Box<Expression>, Box<Expression>),
    Max(Box<Expression>, Box<Expression>),
}

impl Expression {
    /// Evaluate expression with given feature values
    /// Mathematical form: f(x) → ℝ
    pub fn evaluate(&self, features: &[f64]) -> f64 {
        match self {
            Expression::Variable(idx) => {
                features.get(*idx).copied().unwrap_or(0.0)
            },
            Expression::Constant(val) => *val,
            Expression::Add(a, b) => {
                a.evaluate(features) + b.evaluate(features)
            },
            Expression::Subtract(a, b) => {
                a.evaluate(features) - b.evaluate(features)
            },
            Expression::Multiply(a, b) => {
                a.evaluate(features) * b.evaluate(features)
            },
            Expression::Divide(a, b) => {
                let denominator = b.evaluate(features);
                if denominator.abs() < 1e-10 {
                    0.0  // Protected division
                } else {
                    a.evaluate(features) / denominator
                }
            },
            Expression::Sin(a) => {
                let val = a.evaluate(features);
                if val.abs() < 100.0 { val.sin() } else { 0.0 }
            },
            Expression::Cos(a) => {
                let val = a.evaluate(features);
                if val.abs() < 100.0 { val.cos() } else { 0.0 }
            },
            Expression::Log(a) => {
                let val = a.evaluate(features);
                if val > 1e-10 { val.ln() } else { 0.0 }
            },
            Expression::Exp(a) => {
                let val = a.evaluate(features);
                if val < 50.0 { val.exp() } else { 0.0 }  // Prevent overflow
            },
            Expression::Sqrt(a) => {
                let val = a.evaluate(features);
                if val >= 0.0 { val.sqrt() } else { 0.0 }
            },
            Expression::Abs(a) => {
                a.evaluate(features).abs()
            },
            Expression::Tanh(a) => {
                let val = a.evaluate(features);
                if val.abs() < 50.0 { val.tanh() } else { 0.0 }
            },
            Expression::Sigmoid(a) => {
                let val = a.evaluate(features);
                if val.abs() < 50.0 { 1.0 / (1.0 + (-val).exp()) } else { 0.0 }
            },
            Expression::Sign(a) => {
                let val = a.evaluate(features);
                if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 }
            },
            Expression::Power(a, b) => {
                let base = a.evaluate(features);
                let exp = b.evaluate(features);
                if base.abs() < 100.0 && exp.abs() < 10.0 {
                    base.powf(exp)
                } else {
                    0.0  // Protected power
                }
            },
            Expression::Modulo(a, b) => {
                let dividend = a.evaluate(features);
                let divisor = b.evaluate(features);
                if divisor.abs() > 1e-10 {
                    dividend % divisor
                } else {
                    0.0  // Protected modulo
                }
            },
            Expression::Min(a, b) => {
                let val_a = a.evaluate(features);
                let val_b = b.evaluate(features);
                val_a.min(val_b)
            },
            Expression::Max(a, b) => {
                let val_a = a.evaluate(features);
                let val_b = b.evaluate(features);
                val_a.max(val_b)
            },
        }
    }
    
    /// Calculate expression complexity
    /// Mathematical form: complexity(f) = |nodes(f)| + Σ_{op∈f} weight(op)
    pub fn complexity(&self) -> usize {
        match self {
            Expression::Variable(_) | Expression::Constant(_) => 1,
            Expression::Add(a, b) | Expression::Subtract(a, b) | 
            Expression::Multiply(a, b) | Expression::Divide(a, b) |
            Expression::Power(a, b) => {
                1 + a.complexity() + b.complexity()
            },
            Expression::Sin(a) | Expression::Cos(a) | Expression::Log(a) | 
            Expression::Exp(a) | Expression::Sqrt(a) | Expression::Abs(a) |
            Expression::Tanh(a) | Expression::Sigmoid(a) | Expression::Sign(a) => {
                2 + a.complexity()  // Unary operations have higher weight
            },
            Expression::Modulo(a, b) | Expression::Min(a, b) | Expression::Max(a, b) => {
                2 + a.complexity() + b.complexity()  // Binary operations with higher weight
            },
        }
    }
    
    /// Generate random expression tree
    pub fn random(max_depth: usize, num_variables: usize) -> Self {
        let mut rng = thread_rng();
        Self::random_with_rng(&mut rng, max_depth, num_variables)
    }
    
    /// Generate random expression with given RNG - enhanced for strategic chess patterns
    fn random_with_rng(rng: &mut impl Rng, max_depth: usize, num_variables: usize) -> Self {
        if max_depth == 0 {
            // Terminal nodes only
            if rng.gen_bool(0.6) {
                Expression::Variable(rng.gen_range(0..num_variables))
            } else {
                Expression::Constant(rng.gen_range(-20.0..20.0)) // Wider constant range
            }
        } else {
            // Favor complex operations for chess pattern discovery
            match rng.gen_range(0..15) { // More diverse operation selection
                0..=2 => {
                    // Basic binary operations
                    let left = Box::new(Self::random_with_rng(rng, max_depth - 1, num_variables));
                    let right = Box::new(Self::random_with_rng(rng, max_depth - 1, num_variables));
                    
                    match rng.gen_range(0..4) {
                        0 => Expression::Add(left, right),
                        1 => Expression::Subtract(left, right),
                        2 => Expression::Multiply(left, right),
                        3 => Expression::Divide(left, right),
                        _ => Expression::Add(left, right),
                    }
                },
                3..=5 => {
                    // Strategic unary operations (higher probability)
                    let child = Box::new(Self::random_with_rng(rng, max_depth - 1, num_variables));
                    match rng.gen_range(0..9) {
                        0 => Expression::Tanh(child),    // Bounded activation
                        1 => Expression::Sigmoid(child), // Decision boundary
                        2 => Expression::Sign(child),    // Discrete decisions
                        3 => Expression::Abs(child),     // Magnitude
                        4 => Expression::Sin(child),     // Periodic patterns
                        5 => Expression::Cos(child),     // Periodic patterns
                        6 => Expression::Log(child),     // Logarithmic scaling
                        7 => Expression::Exp(child),     // Exponential growth
                        8 => Expression::Sqrt(child),    // Square root scaling
                        _ => Expression::Tanh(child),
                    }
                },
                6..=8 => {
                    // Strategic binary operations (higher probability)
                    let left = Box::new(Self::random_with_rng(rng, max_depth - 1, num_variables));
                    let right = Box::new(Self::random_with_rng(rng, max_depth - 1, num_variables));
                    
                    match rng.gen_range(0..5) {
                        0 => Expression::Min(left, right),    // Strategic choices
                        1 => Expression::Max(left, right),    // Strategic choices
                        2 => Expression::Modulo(left, right), // Cyclic patterns
                        3 => Expression::Power(left, right),  // Non-linear relationships
                        4 => Expression::Multiply(left, right), // Interactions
                        _ => Expression::Max(left, right),
                    }
                },
                9..=10 => {
                    // Power operations for non-linear patterns
                    let left = Box::new(Self::random_with_rng(rng, max_depth - 1, num_variables));
                    let right = Box::new(Expression::Constant(rng.gen_range(1.0..4.0))); // Small integer powers
                    Expression::Power(left, right)
                },
                11..=12 => {
                    // Complex compositions (only if sufficient depth)
                    if max_depth >= 2 {
                        let inner = Box::new(Self::random_with_rng(rng, max_depth - 2, num_variables));
                        match rng.gen_range(0..3) {
                            0 => Expression::Sigmoid(Box::new(Expression::Multiply(inner, Box::new(Expression::Constant(rng.gen_range(0.1..5.0)))))),
                            1 => Expression::Tanh(Box::new(Expression::Add(inner, Box::new(Expression::Constant(rng.gen_range(-2.0..2.0)))))),
                            2 => Expression::Max(inner, Box::new(Expression::Constant(0.0))), // ReLU-like
                            _ => Expression::Sigmoid(inner),
                        }
                    } else {
                        // Fall back to simple terminal
                        if rng.gen_bool(0.5) {
                            Expression::Variable(rng.gen_range(0..num_variables))
                        } else {
                            Expression::Constant(rng.gen_range(-10.0..10.0))
                        }
                    }
                },
                _ => {
                    // Terminal nodes
                    if rng.gen_bool(0.5) {
                        Expression::Variable(rng.gen_range(0..num_variables))
                    } else {
                        Expression::Constant(rng.gen_range(-10.0..10.0))
                    }
                }
            }
        }
    }
    
    /// Mutate expression tree
    pub fn mutate(&mut self, mutation_rate: f64, num_variables: usize) {
        let mut rng = thread_rng();
        self.mutate_with_rng(&mut rng, mutation_rate, num_variables);
    }
    
    /// Mutate expression with given RNG
    fn mutate_with_rng(&mut self, rng: &mut impl Rng, mutation_rate: f64, num_variables: usize) {
        if rng.gen::<f64>() < mutation_rate {
            // Mutate this node
            match self {
                Expression::Variable(idx) => {
                    *idx = rng.gen_range(0..num_variables);
                },
                Expression::Constant(val) => {
                    *val += rng.gen_range(-1.0..1.0);
                },
                _ => {
                    // Replace with new random subtree
                    *self = Self::random_with_rng(rng, 3, num_variables);
                }
            }
        } else {
            // Recurse into children
            match self {
                Expression::Add(a, b) | Expression::Subtract(a, b) | 
                Expression::Multiply(a, b) | Expression::Divide(a, b) |
                Expression::Power(a, b) => {
                    a.mutate_with_rng(rng, mutation_rate, num_variables);
                    b.mutate_with_rng(rng, mutation_rate, num_variables);
                },
                Expression::Sin(a) | Expression::Cos(a) | Expression::Log(a) | 
                Expression::Exp(a) | Expression::Sqrt(a) | Expression::Abs(a) |
                Expression::Tanh(a) | Expression::Sigmoid(a) | Expression::Sign(a) => {
                    a.mutate_with_rng(rng, mutation_rate, num_variables);
                },
                Expression::Modulo(a, b) | Expression::Min(a, b) | Expression::Max(a, b) => {
                    a.mutate_with_rng(rng, mutation_rate, num_variables);
                    b.mutate_with_rng(rng, mutation_rate, num_variables);
                },
                _ => {} // Terminal nodes don't recurse
            }
        }
    }
    
    /// Crossover with another expression
    pub fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        self.crossover_with_rng(other, &mut rng)
    }
    
    /// Crossover with given RNG
    fn crossover_with_rng(&self, other: &Self, rng: &mut impl Rng) -> Self {
        if rng.gen_bool(0.1) {
            // Swap entire subtrees
            other.clone()
        } else {
            // Recurse and crossover children
            match (self, other) {
                (Expression::Add(a1, b1), Expression::Add(a2, b2)) => {
                    Expression::Add(
                        Box::new(a1.crossover_with_rng(a2, rng)),
                        Box::new(b1.crossover_with_rng(b2, rng))
                    )
                },
                (Expression::Multiply(a1, b1), Expression::Multiply(a2, b2)) => {
                    Expression::Multiply(
                        Box::new(a1.crossover_with_rng(a2, rng)),
                        Box::new(b1.crossover_with_rng(b2, rng))
                    )
                },
                (Expression::Sin(a1), Expression::Sin(a2)) => {
                    Expression::Sin(Box::new(a1.crossover_with_rng(a2, rng)))
                },
                _ => self.clone()  // Different types, keep original
            }
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Variable(idx) => write!(f, "x{}", idx),
            Expression::Constant(val) => write!(f, "{:.3}", val),
            Expression::Add(a, b) => write!(f, "({} + {})", a, b),
            Expression::Subtract(a, b) => write!(f, "({} - {})", a, b),
            Expression::Multiply(a, b) => write!(f, "({} * {})", a, b),
            Expression::Divide(a, b) => write!(f, "({} / {})", a, b),
            Expression::Sin(a) => write!(f, "sin({})", a),
            Expression::Cos(a) => write!(f, "cos({})", a),
            Expression::Log(a) => write!(f, "log({})", a),
            Expression::Exp(a) => write!(f, "exp({})", a),
            Expression::Sqrt(a) => write!(f, "sqrt({})", a),
            Expression::Abs(a) => write!(f, "abs({})", a),
            Expression::Tanh(a) => write!(f, "tanh({})", a),
            Expression::Sigmoid(a) => write!(f, "sigmoid({})", a),
            Expression::Sign(a) => write!(f, "sign({})", a),
            Expression::Power(a, b) => write!(f, "({} ^ {})", a, b),
            Expression::Modulo(a, b) => write!(f, "({} % {})", a, b),
            Expression::Min(a, b) => write!(f, "min({}, {})", a, b),
            Expression::Max(a, b) => write!(f, "max({}, {})", a, b),
        }
    }
}

/// Genetic Programming Parameters
#[derive(Debug, Clone)]
pub struct SymbolicRegressionConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub max_depth: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism_rate: f64,
    pub complexity_penalty: f64,
    pub target_fitness: f64,
}

impl Default for SymbolicRegressionConfig {
    fn default() -> Self {
        Self {
            population_size: 120,        // Larger population for diversity
            max_generations: 60,         // More generations for complex evolution
            max_depth: 7,               // Allow deeper, more complex expressions
            mutation_rate: 0.18,        // Higher mutation for exploration
            crossover_rate: 0.75,       // Balanced crossover
            elitism_rate: 0.12,         // Keep more elite individuals
            complexity_penalty: 0.003,  // Very low penalty to encourage complexity
            target_fitness: 0.85,       // More achievable target for complex patterns
        }
    }
}

/// Symbolic Regression Engine
pub struct SymbolicRegression {
    config: SymbolicRegressionConfig,
    num_variables: usize,
}

impl SymbolicRegression {
    pub fn new(config: SymbolicRegressionConfig, num_variables: usize) -> Self {
        Self { config, num_variables }
    }
    
    /// Evolve population to find best expression
    /// Mathematical form: argmax_f L(f, X, y)
    pub fn evolve(&self, 
                  inputs: &[Array1<f64>], 
                  targets: &[f64]) -> Result<(Expression, f64)> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(anyhow::anyhow!("Empty input data"));
        }
        
        println!("🧬 Starting symbolic regression evolution");
        println!("   Population: {}, Generations: {}", 
                 self.config.population_size, self.config.max_generations);
        
        // Generate initial population
        let mut population = self.generate_initial_population();
        let mut best_fitness = 0.0;
        let mut best_expression = population[0].clone();
        
        for generation in 0..self.config.max_generations {
            // Evaluate fitness for all individuals
            let fitness_scores: Vec<f64> = population.iter()
                .map(|expr| self.evaluate_fitness(expr, inputs, targets))
                .collect();
            
            // Track best individual
            if let Some((best_idx, &current_best)) = fitness_scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
                
                if current_best > best_fitness {
                    best_fitness = current_best;
                    best_expression = population[best_idx].clone();
                    
                    if generation % 10 == 0 {
                        println!("   Generation {}: Best fitness = {:.6}, Expression = {}", 
                                 generation, best_fitness, best_expression);
                    }
                }
            }
            
            // Early stopping
            if best_fitness > self.config.target_fitness {
                println!("   🎯 Target fitness reached at generation {}", generation);
                break;
            }
            
            // Evolve population
            population = self.evolve_population(&population, &fitness_scores);
        }
        
        println!("🧬 Evolution complete. Best fitness: {:.6}", best_fitness);
        Ok((best_expression, best_fitness))
    }
    
    /// Generate initial random population
    fn generate_initial_population(&self) -> Vec<Expression> {
        (0..self.config.population_size)
            .map(|_| Expression::random(self.config.max_depth, self.num_variables))
            .collect()
    }
    
    /// Evaluate fitness of expression with strategic chess relevance
    /// Mathematical form: L(f) = R² + strategic_bonus - λ * complexity(f)
    fn evaluate_fitness(&self, expr: &Expression, inputs: &[Array1<f64>], targets: &[f64]) -> f64 {
        let mut sum_squared_error = 0.0;
        let mut sum_targets = 0.0;
        let mut valid_evaluations = 0;
        let mut predictions = Vec::new();
        
        // Calculate MSE and collect predictions
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let prediction = expr.evaluate(input.as_slice().unwrap());
            
            if prediction.is_finite() {
                sum_squared_error += (prediction - target).powi(2);
                sum_targets += target;
                valid_evaluations += 1;
                predictions.push(prediction);
            } else {
                predictions.push(0.0);
            }
        }
        
        if valid_evaluations == 0 {
            return 0.0;
        }
        
        // Calculate R-squared
        let mean_target = sum_targets / valid_evaluations as f64;
        let mut sum_squared_total = 0.0;
        
        for &target in targets.iter() {
            sum_squared_total += (target - mean_target).powi(2);
        }
        
        let r_squared = if sum_squared_total > 0.0 {
            1.0 - (sum_squared_error / sum_squared_total)
        } else {
            0.0
        };
        
        // Strategic relevance bonus for chess-meaningful expressions
        let strategic_bonus = self.calculate_strategic_bonus(expr, &predictions, targets);
        
        // Apply complexity penalty (encourage parsimony)
        let complexity = expr.complexity() as f64;
        let fitness = r_squared + strategic_bonus - self.config.complexity_penalty * complexity;
        
        fitness.max(0.0)
    }
    
    /// Calculate strategic bonus for chess-relevant expressions
    fn calculate_strategic_bonus(&self, expr: &Expression, predictions: &[f64], targets: &[f64]) -> f64 {
        let mut strategic_score = 0.0;
        
        // Bonus for expressions that use chess-strategic operators
        strategic_score += self.count_strategic_operators(expr) * 0.05;
        
        // Bonus for expressions that show decision boundaries (good for win/loss prediction)
        strategic_score += self.evaluate_decision_boundary_quality(predictions) * 0.1;
        
        // Bonus for expressions that correlate with outcome variance
        strategic_score += self.evaluate_outcome_correlation(predictions, targets) * 0.15;
        
        // Penalty for trivial expressions (constants, single variables)
        if self.is_trivial_expression(expr) {
            strategic_score -= 0.2;
        }
        
        strategic_score.max(0.0)
    }
    
    /// Count strategic operators that are meaningful in chess
    fn count_strategic_operators(&self, expr: &Expression) -> f64 {
        match expr {
            Expression::Sigmoid(_) => 1.0, // Decision boundaries
            Expression::Tanh(_) => 1.0,    // Bounded strategic values
            Expression::Min(left, right) | Expression::Max(left, right) => {
                0.5 + self.count_strategic_operators(left) + self.count_strategic_operators(right)
            },
            Expression::Modulo(left, right) => {
                0.3 + self.count_strategic_operators(left) + self.count_strategic_operators(right)
            },
            Expression::Multiply(left, right) | Expression::Add(left, right) |
            Expression::Subtract(left, right) | Expression::Divide(left, right) => {
                self.count_strategic_operators(left) + self.count_strategic_operators(right)
            },
            Expression::Sin(child) | Expression::Cos(child) | Expression::Log(child) |
            Expression::Exp(child) | Expression::Sqrt(child) | Expression::Abs(child) |
            Expression::Sign(child) => {
                self.count_strategic_operators(child)
            },
            _ => 0.0,
        }
    }
    
    /// Evaluate quality of decision boundaries for win/loss prediction
    fn evaluate_decision_boundary_quality(&self, predictions: &[f64]) -> f64 {
        if predictions.len() < 3 {
            return 0.0;
        }
        
        // Calculate prediction spread (good for decision making)
        let min_pred = predictions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_pred = predictions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let spread = max_pred - min_pred;
        
        // Bonus for reasonable spread that could distinguish outcomes
        if spread > 0.1 && spread < 10.0 {
            (spread / 10.0).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Evaluate correlation with strategic outcomes
    fn evaluate_outcome_correlation(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if predictions.len() != targets.len() || predictions.len() < 2 {
            return 0.0;
        }
        
        // Calculate correlation coefficient
        let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
        
        let numerator: f64 = predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - mean_pred) * (t - mean_target))
            .sum();
        
        let pred_variance: f64 = predictions.iter()
            .map(|&p| (p - mean_pred).powi(2))
            .sum();
        
        let target_variance: f64 = targets.iter()
            .map(|&t| (t - mean_target).powi(2))
            .sum();
        
        if pred_variance > 0.0 && target_variance > 0.0 {
            let correlation = numerator / (pred_variance * target_variance).sqrt();
            correlation.abs() // Reward both positive and negative correlations
        } else {
            0.0
        }
    }
    
    /// Check if expression is trivial (just constant or single variable)
    fn is_trivial_expression(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Constant(_) => true,
            Expression::Variable(_) => true,
            _ => false,
        }
    }
    
    /// Evolve population through selection, crossover, and mutation
    fn evolve_population(&self, population: &[Expression], fitness_scores: &[f64]) -> Vec<Expression> {
        let mut new_population = Vec::with_capacity(self.config.population_size);
        let mut rng = thread_rng();
        
        // Elitism: keep best individuals
        let num_elite = (self.config.population_size as f64 * self.config.elitism_rate) as usize;
        let mut elite_indices: Vec<usize> = (0..population.len()).collect();
        elite_indices.sort_by(|&a, &b| fitness_scores[b].partial_cmp(&fitness_scores[a]).unwrap());
        
        for &idx in elite_indices.iter().take(num_elite) {
            new_population.push(population[idx].clone());
        }
        
        // Fill rest with offspring
        while new_population.len() < self.config.population_size {
            let parent1 = self.tournament_selection(population, fitness_scores, &mut rng);
            let parent2 = self.tournament_selection(population, fitness_scores, &mut rng);
            
            let mut child = if rng.gen::<f64>() < self.config.crossover_rate {
                parent1.crossover(parent2)
            } else {
                parent1.clone()
            };
            
            child.mutate(self.config.mutation_rate, self.num_variables);
            new_population.push(child);
        }
        
        new_population
    }
    
    /// Tournament selection
    fn tournament_selection<'a>(&self, population: &'a [Expression], fitness_scores: &[f64], rng: &mut impl Rng) -> &'a Expression {
        let tournament_size = 3;
        let mut best_idx = rng.gen_range(0..population.len());
        let mut best_fitness = fitness_scores[best_idx];
        
        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..population.len());
            if fitness_scores[idx] > best_fitness {
                best_idx = idx;
                best_fitness = fitness_scores[idx];
            }
        }
        
        &population[best_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_expression_evaluation() {
        let expr = Expression::Add(
            Box::new(Expression::Variable(0)),
            Box::new(Expression::Constant(5.0))
        );
        
        let features = [3.0, 2.0, 1.0];
        assert_eq!(expr.evaluate(&features), 8.0);
    }
    
    #[test]
    fn test_expression_complexity() {
        let expr = Expression::Add(
            Box::new(Expression::Variable(0)),
            Box::new(Expression::Sin(Box::new(Expression::Variable(1))))
        );
        
        assert_eq!(expr.complexity(), 4); // Add(1) + Variable(1) + Sin(2)
    }
    
    #[test]
    fn test_symbolic_regression() {
        let config = SymbolicRegressionConfig::default();
        let sr = SymbolicRegression::new(config, 2);
        
        // Test data: y = 2*x1 + 3*x2
        let inputs = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![2.0, 3.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];
        let targets = vec![8.0, 13.0, 18.0]; // 2*1 + 3*2 = 8, etc.
        
        let result = sr.evolve(&inputs, &targets);
        assert!(result.is_ok());
        
        let (_expr, fitness) = result.unwrap();
        assert!(fitness > 0.0);
    }
}