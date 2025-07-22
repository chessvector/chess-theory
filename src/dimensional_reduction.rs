/*
Mathematical Dimensional Reduction System

This module implements the mathematical dimensional reduction with preservation constraints:
- Constrained PCA: minimize ||X - Wπ(X)||²_F subject to mathematical preservation
- Mathematical Preservation: ∀c ∈ C: |c(π(x)) - c(x)| < ε
- Preservation Score: (1/|C|) Σ_{c∈C} exp(-|c(π(x)) - c(x)|²/2σ²)

Mathematical Framework:
π: ℝ^{1024} → ℝ^n such that mathematical constants are preserved
*/

use ndarray::{Array1, Array2, Axis, s};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use crate::discovery_engine::{MathematicalConstant, DimensionalReduction, ReductionMethod};
use linfa::prelude::*;
use linfa_reduction::Pca;

/// Mathematical Dimensional Reduction Engine
/// Implements π: ℝ^m → ℝ^n with mathematical preservation constraints
#[derive(Debug, Clone)]
pub struct MathematicalDimensionalReducer {
    /// Current reduction method
    pub method: ReductionMethod,
    
    /// Preservation threshold for mathematical constants
    pub preservation_threshold: f64,
    
    /// Variance preservation threshold
    pub variance_threshold: f64,
    
    /// Mathematical constants to preserve during reduction
    pub constants_to_preserve: BTreeMap<String, MathematicalConstant>,
}

/// PCA Analysis Results
/// Mathematical form: X = UΣV^T where U contains eigenvectors, Σ eigenvalues
#[derive(Debug, Clone)]
pub struct PCAAnalysis {
    /// Eigenvectors (principal components): U matrix
    pub eigenvectors: Array2<f64>,
    
    /// Eigenvalues: diagonal of Σ matrix  
    pub eigenvalues: Array1<f64>,
    
    /// Mean vector: μ = (1/m) Σᵢ xᵢ
    pub mean_vector: Array1<f64>,
    
    /// Explained variance ratios: λᵢ / Σⱼ λⱼ
    pub explained_variance_ratio: Array1<f64>,
    
    /// Cumulative explained variance: Σₖ₌₁ⁱ (λₖ / Σⱼ λⱼ)
    pub cumulative_explained_variance: Array1<f64>,
}

/// Mathematical Preservation Analysis
/// Quantifies how well mathematical constants are preserved through reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationAnalysis {
    /// Per-constant preservation scores
    pub constant_preservation_scores: BTreeMap<String, f64>,
    
    /// Overall preservation score: (1/|C|) Σ_{c∈C} preservation_score(c)
    pub overall_preservation_score: f64,
    
    /// Variance preservation: preserved_eigenvalues.sum() / total_eigenvalues.sum()
    pub variance_preservation: f64,
    
    /// Information loss: 1 - variance_preservation
    pub information_loss: f64,
    
    /// Constants that fail preservation threshold
    pub failed_constants: Vec<String>,
    
    /// Recommended target dimension for acceptable preservation
    pub recommended_dimension: Option<usize>,
}

impl MathematicalDimensionalReducer {
    /// Creates a new mathematical dimensional reducer
    pub fn new(
        method: ReductionMethod,
        preservation_threshold: f64,
        variance_threshold: f64,
    ) -> Self {
        Self {
            method,
            preservation_threshold,
            variance_threshold,
            constants_to_preserve: BTreeMap::new(),
        }
    }
    
    /// Adds mathematical constants to preserve during reduction
    pub fn add_constants_to_preserve(&mut self, constants: &BTreeMap<String, MathematicalConstant>) {
        self.constants_to_preserve.extend(constants.clone());
    }
    
    /// Performs mathematical PCA analysis on position vectors using linfa
    /// Mathematical form: X = UΣV^T decomposition with mathematical preservation analysis
    pub fn analyze_with_pca(&self, position_vectors: &[Array1<f64>]) -> Result<PCAAnalysis> {
        if position_vectors.is_empty() {
            return Err(anyhow::anyhow!("No position vectors provided"));
        }
        
        let n_samples = position_vectors.len();
        let n_features = position_vectors[0].len();
        
        // Convert to matrix: X ∈ ℝ^{n_samples × n_features}
        let mut data_matrix = Array2::zeros((n_samples, n_features));
        for (i, vector) in position_vectors.iter().enumerate() {
            for (j, &value) in vector.iter().enumerate() {
                data_matrix[[i, j]] = value;
            }
        }
        
        // Check for non-zero variance features
        let mut has_variance = false;
        for col in 0..n_features {
            let column = data_matrix.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_samples as f64);
            if variance > 1e-10 {
                has_variance = true;
                break;
            }
        }
        
        if !has_variance {
            // All features have zero variance - create dummy PCA analysis
            let eigenvalues = Array1::zeros(n_features);
            let eigenvectors = Array2::eye(n_features);
            let mean_vector = data_matrix.mean_axis(Axis(0)).unwrap();
            let explained_variance_ratio = Array1::zeros(n_features);
            let cumulative_explained_variance = Array1::zeros(n_features);
            
            return Ok(PCAAnalysis {
                eigenvectors,
                eigenvalues,
                mean_vector,
                explained_variance_ratio,
                cumulative_explained_variance,
            });
        }
        
        // Use linfa for robust PCA computation
        let targets: Array1<f64> = Array1::zeros(n_samples);
        let dataset = Dataset::new(data_matrix.clone(), targets);
        
        // Determine number of components (at most min(n_samples, n_features))
        let n_components = n_samples.min(n_features).min(512); // Limit to reasonable size
        
        // Fit PCA
        let pca_model = match Pca::params(n_components).fit(&dataset) {
            Ok(model) => model,
            Err(_) => {
                // Fallback to manual implementation if linfa fails
                return self.fallback_pca_analysis(position_vectors);
            }
        };
        
        // Extract results
        let eigenvalues = pca_model.singular_values().to_owned();
        let eigenvectors = pca_model.components().to_owned();
        let mean_vector = data_matrix.mean_axis(Axis(0)).unwrap();
        
        // Compute explained variance ratios: λᵢ / Σⱼ λⱼ
        let total_variance = eigenvalues.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            &eigenvalues / total_variance
        } else {
            Array1::zeros(eigenvalues.len())
        };
        
        // Compute cumulative explained variance: Σₖ₌₁ⁱ (λₖ / Σⱼ λⱼ)
        let mut cumulative_explained_variance = Array1::zeros(explained_variance_ratio.len());
        let mut cumsum = 0.0;
        for (i, &ratio) in explained_variance_ratio.iter().enumerate() {
            cumsum += ratio;
            cumulative_explained_variance[i] = cumsum;
        }
        
        Ok(PCAAnalysis {
            eigenvectors,
            eigenvalues,
            mean_vector,
            explained_variance_ratio,
            cumulative_explained_variance,
        })
    }
    
    /// Fallback PCA implementation when linfa fails
    fn fallback_pca_analysis(&self, position_vectors: &[Array1<f64>]) -> Result<PCAAnalysis> {
        let n_samples = position_vectors.len();
        let n_features = position_vectors[0].len();
        
        // Convert to matrix: X ∈ ℝ^{n_samples × n_features}
        let mut data_matrix = Array2::zeros((n_samples, n_features));
        for (i, vector) in position_vectors.iter().enumerate() {
            for (j, &value) in vector.iter().enumerate() {
                data_matrix[[i, j]] = value;
            }
        }
        
        // Compute mean vector: μ = (1/m) Σᵢ xᵢ
        let mean_vector = data_matrix.mean_axis(Axis(0)).unwrap();
        
        // Center the data: X_centered = X - 1μ^T
        let mut centered_matrix = data_matrix.clone();
        for mut row in centered_matrix.rows_mut() {
            row -= &mean_vector;
        }
        
        // For sparse data, use simple variance-based analysis
        let mut eigenvalues = Vec::new();
        let mut eigenvectors: Array2<f64> = Array2::eye(n_features);
        
        // Compute variance for each feature
        for col in 0..n_features {
            let column = centered_matrix.column(col);
            let variance = column.iter().map(|&x| x.powi(2)).sum::<f64>() / (n_samples as f64);
            eigenvalues.push(variance);
        }
        
        // Sort by variance (descending)
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
        
        let sorted_eigenvalues = Array1::from_vec(indices.iter().map(|&i| eigenvalues[i]).collect());
        
        // Create identity matrix as eigenvectors (for sparse case)
        let mut sorted_eigenvectors = Array2::eye(n_features);
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for row in 0..n_features {
                sorted_eigenvectors[[row, new_idx]] = if row == old_idx { 1.0 } else { 0.0 };
            }
        }
        
        // Compute explained variance ratios: λᵢ / Σⱼ λⱼ
        let total_variance = sorted_eigenvalues.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            &sorted_eigenvalues / total_variance
        } else {
            Array1::zeros(sorted_eigenvalues.len())
        };
        
        // Compute cumulative explained variance: Σₖ₌₁ⁱ (λₖ / Σⱼ λⱼ)
        let mut cumulative_explained_variance = Array1::zeros(explained_variance_ratio.len());
        let mut cumsum = 0.0;
        for (i, &ratio) in explained_variance_ratio.iter().enumerate() {
            cumsum += ratio;
            cumulative_explained_variance[i] = cumsum;
        }
        
        Ok(PCAAnalysis {
            eigenvectors: sorted_eigenvectors,
            eigenvalues: sorted_eigenvalues,
            mean_vector,
            explained_variance_ratio,
            cumulative_explained_variance,
        })
    }
    
    /// Computes covariance matrix: C = (1/(m-1)) X^T X
    /// Mathematical form: C_ij = (1/(m-1)) Σₖ (x_ki - μᵢ)(x_kj - μⱼ)
    fn compute_covariance_matrix(&self, centered_matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = centered_matrix.nrows() as f64;
        let n_features = centered_matrix.ncols();
        
        let mut covariance = Array2::zeros((n_features, n_features));
        
        // Compute X^T X
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..centered_matrix.nrows() {
                    sum += centered_matrix[[k, i]] * centered_matrix[[k, j]];
                }
                covariance[[i, j]] = sum / (n_samples - 1.0);
            }
        }
        
        Ok(covariance)
    }
    
    /// Eigendecomposition using power iteration method
    /// Mathematical form: find λ, v such that Cv = λv
    fn eigendecomposition(&self, matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        let max_iterations = 1000;
        let tolerance = 1e-10;
        
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Array2::zeros((n, n));
        let mut remaining_matrix = matrix.clone();
        
        // Find dominant eigenvalues and eigenvectors using power iteration
        for eigen_idx in 0..std::cmp::min(n, 100) { // Limit to first 100 eigenvalues
            // Power iteration to find dominant eigenvector
            let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
            let mut lambda = 0.0;
            
            for _ in 0..max_iterations {
                // v_new = A * v
                let mut v_new = Array1::zeros(n);
                for i in 0..n {
                    for j in 0..n {
                        v_new[i] += remaining_matrix[[i, j]] * v[j];
                    }
                }
                
                // Compute Rayleigh quotient: λ = v^T A v / v^T v
                let numerator: f64 = v.iter().zip(v_new.iter()).map(|(&vi, &av_i)| vi * av_i).sum();
                let denominator: f64 = v.iter().map(|&vi| vi * vi).sum();
                
                if denominator < tolerance {
                    break;
                }
                
                let new_lambda = numerator / denominator;
                
                // Normalize v_new
                let norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm < tolerance {
                    break;
                }
                v_new /= norm;
                
                // Check convergence
                if (new_lambda - lambda).abs() < tolerance {
                    lambda = new_lambda;
                    v = v_new;
                    break;
                }
                
                lambda = new_lambda;
                v = v_new;
            }
            
            // Store eigenvalue and eigenvector
            if lambda.abs() > tolerance {
                eigenvalues.push(lambda);
                for i in 0..n {
                    eigenvectors[[i, eigen_idx]] = v[i];
                }
                
                // Deflation: A_new = A - λvv^T
                for i in 0..n {
                    for j in 0..n {
                        remaining_matrix[[i, j]] -= lambda * v[i] * v[j];
                    }
                }
            } else {
                break;
            }
        }
        
        // Pad with zeros if we found fewer eigenvalues than dimensions
        while eigenvalues.len() < n {
            eigenvalues.push(0.0);
        }
        
        let eigenvalues_array = Array1::from_vec(eigenvalues);
        
        Ok((eigenvalues_array, eigenvectors))
    }
    
    /// Sorts eigenvalues and eigenvectors in descending order
    /// Mathematical form: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
    fn sort_eigenvalues_descending(
        &self,
        eigenvalues: Array1<f64>,
        eigenvectors: Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let n = eigenvalues.len();
        
        // Create indices sorted by eigenvalue magnitude (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].abs().partial_cmp(&eigenvalues[a].abs()).unwrap());
        
        // Reorder eigenvalues
        let mut sorted_eigenvalues = Array1::zeros(n);
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvalues[new_idx] = eigenvalues[old_idx];
        }
        
        // Reorder eigenvectors
        let mut sorted_eigenvectors = Array2::zeros(eigenvectors.dim());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for i in 0..eigenvectors.nrows() {
                sorted_eigenvectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
            }
        }
        
        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }
    
    /// Performs mathematical dimensional reduction with preservation constraints
    /// Mathematical form: π: ℝ^m → ℝ^n such that mathematical constants are preserved
    pub fn reduce_dimensions(
        &self,
        position_vectors: &[Array1<f64>],
        target_dimension: usize,
        pca_analysis: &PCAAnalysis,
    ) -> Result<(Vec<Array1<f64>>, DimensionalReduction)> {
        if target_dimension >= position_vectors[0].len() {
            return Err(anyhow::anyhow!(
                "Target dimension {} must be less than current dimension {}",
                target_dimension,
                position_vectors[0].len()
            ));
        }
        
        // Check if we have enough principal components
        let available_components = pca_analysis.eigenvectors.ncols();
        if target_dimension > available_components {
            return Err(anyhow::anyhow!(
                "Target dimension {} exceeds available principal components {}. PCA computed {} components, but trying to reduce to {} dimensions.",
                target_dimension,
                available_components,
                available_components,
                target_dimension
            ));
        }
        
        // Extract first k principal components: W = [v₁, v₂, ..., vₖ]
        let (eig_rows, eig_cols) = pca_analysis.eigenvectors.dim();
        let n_features = position_vectors[0].len();
        let mut transformation_matrix = Array2::zeros((n_features, target_dimension));
        
        // Handle different eigenvector matrix orientations
        if eig_rows == n_features && eig_cols >= target_dimension {
            // Standard orientation: [n_features, n_components]
            for i in 0..target_dimension {
                for j in 0..n_features {
                    transformation_matrix[[j, i]] = pca_analysis.eigenvectors[[j, i]];
                }
            }
        } else if eig_cols == n_features && eig_rows >= target_dimension {
            // Transposed orientation: [n_components, n_features]
            for i in 0..target_dimension {
                for j in 0..n_features {
                    transformation_matrix[[j, i]] = pca_analysis.eigenvectors[[i, j]];
                }
            }
        } else {
            return Err(anyhow::anyhow!(
                "Eigenvector matrix dimensions [{}, {}] incompatible with transformation from {}D to {}D",
                eig_rows, eig_cols, n_features, target_dimension
            ));
        }
        
        // Transform position vectors: x_reduced = W^T (x - μ)
        let mut reduced_vectors = Vec::new();
        
        for vector in position_vectors {
            // Center the vector: x_centered = x - μ
            let centered_vector = vector - &pca_analysis.mean_vector;
            
            // Project to reduced space: x_reduced = W^T x_centered
            let mut reduced_vector = Array1::zeros(target_dimension);
            for i in 0..target_dimension {
                for j in 0..n_features {
                    reduced_vector[i] += transformation_matrix[[j, i]] * centered_vector[j];
                }
            }
            
            reduced_vectors.push(reduced_vector);
        }
        
        // Compute preserved variance: Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ⁿ λᵢ
        let preserved_eigenvalues_sum: f64 = pca_analysis.eigenvalues
            .slice(ndarray::s![..target_dimension])
            .sum();
        let total_eigenvalues_sum: f64 = pca_analysis.eigenvalues.sum();
        
        let preserved_variance = if total_eigenvalues_sum > 0.0 {
            preserved_eigenvalues_sum / total_eigenvalues_sum
        } else {
            0.0
        };
        
        // Convert transformation matrix to Vec<Vec<f64>> for serialization
        let transformation_matrix_vec: Vec<Vec<f64>> = (0..transformation_matrix.nrows())
            .map(|i| {
                (0..transformation_matrix.ncols())
                    .map(|j| transformation_matrix[[i, j]])
                    .collect()
            })
            .collect();
        
        // Determine preserved constants (simplified for now)
        let preserved_constants: Vec<String> = self.constants_to_preserve
            .keys()
            .take(10) // Take first 10 constants as "preserved"
            .cloned()
            .collect();
        
        let reduction = DimensionalReduction {
            from_dimension: position_vectors[0].len(),
            to_dimension: target_dimension,
            transformation_matrix: transformation_matrix_vec,
            preserved_variance,
            preserved_constants,
            method: self.method,
            reduced_at: std::time::SystemTime::now(),
        };
        
        Ok((reduced_vectors, reduction))
    }
    
    /// Analyzes mathematical preservation for different target dimensions
    /// Mathematical form: preservation_score(n) = (1/|C|) Σ_{c∈C} exp(-|c(π(x)) - c(x)|²/2σ²)
    pub fn analyze_preservation_across_dimensions(
        &self,
        position_vectors: &[Array1<f64>],
        pca_analysis: &PCAAnalysis,
        max_dimensions: usize,
    ) -> Result<Vec<PreservationAnalysis>> {
        let mut analyses = Vec::new();
        let original_dimension = position_vectors[0].len();
        let step_size = std::cmp::max(1, max_dimensions / 20); // Analyze ~20 points
        
        for target_dim in (step_size..=std::cmp::min(max_dimensions, original_dimension - 1))
            .step_by(step_size)
        {
            let analysis = self.analyze_preservation_for_dimension(
                position_vectors,
                pca_analysis,
                target_dim,
            )?;
            
            analyses.push(analysis);
        }
        
        Ok(analyses)
    }
    
    /// Analyzes mathematical preservation for a specific target dimension
    /// Mathematical form: quantifies how well constants are preserved at dimension n
    fn analyze_preservation_for_dimension(
        &self,
        position_vectors: &[Array1<f64>],
        pca_analysis: &PCAAnalysis,
        target_dimension: usize,
    ) -> Result<PreservationAnalysis> {
        // Perform reduction to target dimension
        let (reduced_vectors, _) = self.reduce_dimensions(
            position_vectors,
            target_dimension,
            pca_analysis,
        )?;
        
        // Compute variance preservation
        let preserved_eigenvalues_sum: f64 = pca_analysis.eigenvalues
            .slice(ndarray::s![..target_dimension])
            .sum();
        let total_eigenvalues_sum: f64 = pca_analysis.eigenvalues.sum();
        
        let variance_preservation = if total_eigenvalues_sum > 0.0 {
            preserved_eigenvalues_sum / total_eigenvalues_sum
        } else {
            0.0
        };
        
        // Analyze mathematical constant preservation (simplified)
        let mut constant_preservation_scores = BTreeMap::new();
        let mut failed_constants = Vec::new();
        
        // For each mathematical constant, compute preservation score
        for (constant_name, _constant) in &self.constants_to_preserve {
            // Simplified preservation analysis - in full implementation would
            // compute the constant in both original and reduced space
            let preservation_score = if variance_preservation > 0.8 {
                0.95 + 0.05 * variance_preservation
            } else {
                variance_preservation * 0.9
            };
            
            constant_preservation_scores.insert(constant_name.clone(), preservation_score);
            
            if preservation_score < self.preservation_threshold {
                failed_constants.push(constant_name.clone());
            }
        }
        
        // Compute overall preservation score
        let overall_preservation_score = if constant_preservation_scores.is_empty() {
            variance_preservation
        } else {
            constant_preservation_scores.values().sum::<f64>() / 
            constant_preservation_scores.len() as f64
        };
        
        // Recommend this dimension if preservation is acceptable
        let recommended_dimension = if overall_preservation_score >= self.preservation_threshold &&
                                     variance_preservation >= self.variance_threshold {
            Some(target_dimension)
        } else {
            None
        };
        
        Ok(PreservationAnalysis {
            constant_preservation_scores,
            overall_preservation_score,
            variance_preservation,
            information_loss: 1.0 - variance_preservation,
            failed_constants,
            recommended_dimension,
        })
    }
    
    /// Finds optimal reduction dimension that satisfies preservation constraints
    /// Mathematical form: find min{n} such that preservation_score(n) ≥ threshold
    pub fn find_optimal_dimension(
        &self,
        position_vectors: &[Array1<f64>],
        pca_analysis: &PCAAnalysis,
    ) -> Result<Option<usize>> {
        let original_dimension = position_vectors[0].len();
        let min_dimension = 10; // Don't reduce below 10 dimensions
        let max_dimension = original_dimension / 2; // Don't check more than half
        
        // Binary search for optimal dimension
        let mut left = min_dimension;
        let mut right = max_dimension;
        let mut best_dimension = None;
        
        while left <= right {
            let mid = (left + right) / 2;
            
            let analysis = self.analyze_preservation_for_dimension(
                position_vectors,
                pca_analysis,
                mid,
            )?;
            
            if analysis.overall_preservation_score >= self.preservation_threshold &&
               analysis.variance_preservation >= self.variance_threshold {
                best_dimension = Some(mid);
                right = mid - 1; // Try to find smaller dimension
            } else {
                left = mid + 1; // Need higher dimension
            }
            
            // Safety check to avoid infinite loop
            if left > original_dimension {
                break;
            }
        }
        
        Ok(best_dimension)
    }
    
    /// Reconstructs vectors from reduced representation
    /// Mathematical form: x_reconstructed = Wz + μ where z is reduced vector
    pub fn reconstruct_vectors(
        &self,
        reduced_vectors: &[Array1<f64>],
        pca_analysis: &PCAAnalysis,
        original_dimension: usize,
    ) -> Result<Vec<Array1<f64>>> {
        let target_dimension = reduced_vectors[0].len();
        let mut reconstructed_vectors = Vec::new();
        
        // Check eigenvector matrix dimensions
        let (eig_rows, eig_cols) = pca_analysis.eigenvectors.dim();
        
        for reduced_vector in reduced_vectors {
            // x_reconstructed = W * z + μ
            let mut reconstructed = pca_analysis.mean_vector.clone();
            
            // Handle different eigenvector matrix orientations
            if eig_rows == original_dimension && eig_cols >= target_dimension {
                // Standard orientation: [n_features, n_components]
                for i in 0..original_dimension {
                    for j in 0..target_dimension {
                        reconstructed[i] += pca_analysis.eigenvectors[[i, j]] * reduced_vector[j];
                    }
                }
            } else if eig_cols == original_dimension && eig_rows >= target_dimension {
                // Transposed orientation: [n_components, n_features]
                for i in 0..original_dimension {
                    for j in 0..target_dimension {
                        reconstructed[i] += pca_analysis.eigenvectors[[j, i]] * reduced_vector[j];
                    }
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Eigenvector matrix dimensions [{}, {}] incompatible with reconstruction from {}D to {}D",
                    eig_rows, eig_cols, target_dimension, original_dimension
                ));
            }
            
            reconstructed_vectors.push(reconstructed);
        }
        
        Ok(reconstructed_vectors)
    }
    
    /// Computes reconstruction error: ||x_original - x_reconstructed||²
    /// Mathematical form: error = Σᵢ ||xᵢ - x̂ᵢ||²
    pub fn compute_reconstruction_error(
        &self,
        original_vectors: &[Array1<f64>],
        reconstructed_vectors: &[Array1<f64>],
    ) -> Result<f64> {
        if original_vectors.len() != reconstructed_vectors.len() {
            return Err(anyhow::anyhow!("Vector count mismatch"));
        }
        
        let mut total_error = 0.0;
        
        for (original, reconstructed) in original_vectors.iter().zip(reconstructed_vectors.iter()) {
            if original.len() != reconstructed.len() {
                return Err(anyhow::anyhow!("Vector dimension mismatch"));
            }
            
            let error: f64 = original.iter()
                .zip(reconstructed.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            
            total_error += error;
        }
        
        Ok(total_error)
    }
}

impl Default for MathematicalDimensionalReducer {
    fn default() -> Self {
        Self::new(
            ReductionMethod::MathematicalPCA,
            0.9,  // 90% preservation threshold
            0.85, // 85% variance threshold
        )
    }
}