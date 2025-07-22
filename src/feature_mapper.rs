/*
Feature Mapper for Chess Mathematical Discovery Engine

This module provides human-readable interpretations of the 1024-dimensional
chess position vector features.

Vector Structure:
- Dimensions 0-767: Piece positions (12 pieces × 64 squares)
- Dimensions 768-1023: Strategic evaluations and derived features
*/

use crate::{Color, PieceType};

pub struct FeatureMapper {
    feature_descriptions: Vec<String>,
}

impl FeatureMapper {
    pub fn new() -> Self {
        let mut descriptions = Vec::with_capacity(1024);
        
        // Part 1: Piece positions (0-767)
        // 12 piece types × 64 squares = 768 dimensions
        let piece_types = [
            ("white_pawn", PieceType::Pawn, Color::White),
            ("black_pawn", PieceType::Pawn, Color::Black),
            ("white_knight", PieceType::Knight, Color::White),
            ("black_knight", PieceType::Knight, Color::Black),
            ("white_bishop", PieceType::Bishop, Color::White),
            ("black_bishop", PieceType::Bishop, Color::Black),
            ("white_rook", PieceType::Rook, Color::White),
            ("black_rook", PieceType::Rook, Color::Black),
            ("white_queen", PieceType::Queen, Color::White),
            ("black_queen", PieceType::Queen, Color::Black),
            ("white_king", PieceType::King, Color::White),
            ("black_king", PieceType::King, Color::Black),
        ];
        
        for (piece_name, _, _) in &piece_types {
            for square in 0..64 {
                let file = (square % 8) as u8;
                let rank = (square / 8) + 1;
                let square_name = format!("{}{}", 
                    (b'a' + file) as char, 
                    rank);
                descriptions.push(format!("{}_{}", piece_name, square_name));
            }
        }
        
        // Part 2: Enhanced Strategic evaluations (768-799) - Expanded for better pattern discovery
        let strategic_features = vec![
            "material_balance",           // 768 - ∑ material_value_i for i ∈ pieces
            "positional_score",           // 769 - ∑ position_value_i for i ∈ squares
            "white_king_safety",          // 770 - safety_function(white_king)
            "black_king_safety",          // 771 - safety_function(black_king)
            "center_control",             // 772 - ∑ control_value_i for i ∈ center_squares
            "development",                // 773 - measure of piece development
            "pawn_structure",             // 774 - pawn formation evaluation
            "white_to_move",              // 775 - binary: 1 if white to move, -1 if black
            "white_kingside_castle",      // 776 - castling rights
            "white_queenside_castle",     // 777 - castling rights
            "black_kingside_castle",      // 778 - castling rights
            "black_queenside_castle",     // 779 - castling rights
            "en_passant_file",            // 780 - en passant target file
            "en_passant_rank",            // 781 - en passant target rank
            "halfmove_clock",             // 782 - halfmove clock (normalized)
            "fullmove_number",            // 783 - fullmove number (logarithmic)
            // Enhanced strategic features for better pattern discovery
            "material_to_positional_ratio", // 784 - material_balance / positional_score
            "king_safety_differential",     // 785 - white_king_safety - black_king_safety
            "tempo_advantage",              // 786 - development advantage adjusted by turn
            "central_dominance",            // 787 - center_control weighted by material
            "pawn_structure_asymmetry",     // 788 - pawn imbalances between sides
            "piece_activity_white",         // 789 - sum of piece mobility for white
            "piece_activity_black",         // 790 - sum of piece mobility for black  
            "tactical_pressure",            // 791 - pins, forks, discovered attacks
            "space_advantage",              // 792 - territorial control metric
            "initiative_measure",           // 793 - forcing moves and threats
            "endgame_coefficient",          // 794 - transition to endgame indicator
            "attack_potential_white",       // 795 - attacking chances for white
            "attack_potential_black",       // 796 - attacking chances for black
            "positional_tension",           // 797 - unstable tactical situations
            "strategic_balance",            // 798 - overall strategic equilibrium
            "phase_transition",             // 799 - game phase (opening/middle/end)
        ];
        
        for feature in strategic_features {
            descriptions.push(feature.to_string());
        }
        
        // Part 3: Advanced features (800-879) - Tactical and positional details
        // Piece activities (64 features)
        for i in 0..64 {
            descriptions.push(format!("square_activity_{}", i));
        }
        
        // Pawn structure details (8 features)
        for feature in ["pawn_islands_white", "pawn_islands_black", 
                       "doubled_pawns_white", "doubled_pawns_black",
                       "isolated_pawns_white", "isolated_pawns_black",
                       "passed_pawns_white", "passed_pawns_black"] {
            descriptions.push(feature.to_string());
        }
        
        // Piece coordination (4 features)
        for feature in ["piece_protection_white", "piece_protection_black",
                       "piece_attacks_white", "piece_attacks_black"] {
            descriptions.push(feature.to_string());
        }
        
        // Key square control (8 features: 4 central squares × 2 colors)
        for square in ["e4", "e5", "d4", "d5"] {
            descriptions.push(format!("white_control_{}", square));
            descriptions.push(format!("black_control_{}", square));
        }
        
        // Derived mathematical features (fill remaining to 1024)
        while descriptions.len() < 1024 {
            let index = descriptions.len();
            let feature_type = match index % 10 {
                0 => "sin_transform",
                1 => "cos_transform", 
                2 => "tanh_transform",
                3 => "exp_ln_transform",
                4 => "polynomial_transform",
                5 => "sqrt_transform",
                6 => "sign_transform",
                7 => "abs_diff_transform",
                8 => "product_interaction",
                9 => "average_transform",
                _ => "unknown_transform",
            };
            descriptions.push(format!("derived_{}_{}", feature_type, index));
        }
        
        Self { feature_descriptions: descriptions }
    }
    
    /// Get human-readable description of a feature
    pub fn describe_feature(&self, index: usize) -> &str {
        self.feature_descriptions.get(index)
            .map(|s| s.as_str())
            .unwrap_or("unknown_feature")
    }
    
    /// Get the feature category (piece_position, strategic, advanced, derived)
    pub fn get_feature_category(&self, index: usize) -> FeatureCategory {
        match index {
            0..=767 => FeatureCategory::PiecePosition,
            768..=799 => FeatureCategory::Strategic,    // Expanded strategic range
            800..=879 => FeatureCategory::Advanced,
            880..=1023 => FeatureCategory::Derived,
            _ => FeatureCategory::Unknown,
        }
    }
    
    /// Check if a feature represents a piece position
    pub fn is_piece_position(&self, index: usize) -> bool {
        index < 768
    }
    
    /// Check if a feature is strategic (non-positional)
    pub fn is_strategic(&self, index: usize) -> bool {
        index >= 768 && index <= 799  // Updated to match expanded strategic range
    }
    
    /// Get square name from piece position index
    pub fn get_square_from_piece_index(&self, index: usize) -> Option<String> {
        if index >= 768 {
            return None;
        }
        
        let square_within_piece = index % 64;
        let file = (square_within_piece % 8) as u8;
        let rank = (square_within_piece / 8) + 1;
        
        Some(format!("{}{}", (b'a' + file) as char, rank))
    }
    
    /// Get piece type from piece position index
    pub fn get_piece_from_index(&self, index: usize) -> Option<(PieceType, Color)> {
        if index >= 768 {
            return None;
        }
        
        let piece_type_index = index / 64;
        match piece_type_index {
            0 => Some((PieceType::Pawn, Color::White)),
            1 => Some((PieceType::Pawn, Color::Black)),
            2 => Some((PieceType::Knight, Color::White)),
            3 => Some((PieceType::Knight, Color::Black)),
            4 => Some((PieceType::Bishop, Color::White)),
            5 => Some((PieceType::Bishop, Color::Black)),
            6 => Some((PieceType::Rook, Color::White)),
            7 => Some((PieceType::Rook, Color::Black)),
            8 => Some((PieceType::Queen, Color::White)),
            9 => Some((PieceType::Queen, Color::Black)),
            10 => Some((PieceType::King, Color::White)),
            11 => Some((PieceType::King, Color::Black)),
            _ => None,
        }
    }
    
    /// Format a linear relationship in chess terms
    pub fn format_linear_relationship(&self, 
                                     x_feature: usize, 
                                     y_feature: usize, 
                                     coefficient: f64, 
                                     intercept: f64,
                                     correlation: f64) -> String {
        let x_desc = self.describe_feature(x_feature);
        let y_desc = self.describe_feature(y_feature);
        
        let x_category = self.get_feature_category(x_feature);
        let y_category = self.get_feature_category(y_feature);
        
        format!("Chess Relationship: {} = {:.6} × {} + {:.6} (r={:.3})\n  Category: {:?} → {:?}", 
                y_desc, coefficient, x_desc, intercept, correlation, x_category, y_category)
    }
    
    /// Format a symbolic expression in chess terms
    pub fn format_symbolic_expression(&self,
                                     x_feature: usize,
                                     y_feature: usize, 
                                     expression: &str,
                                     r_squared: f64,
                                     fitness: f64,
                                     complexity: usize) -> String {
        let x_desc = self.describe_feature(x_feature);
        let y_desc = self.describe_feature(y_feature);
        
        let x_category = self.get_feature_category(x_feature);
        let y_category = self.get_feature_category(y_feature);
        
        format!("Chess Symbolic Discovery: {} = {} (R²={:.3}, fitness={:.3}, complexity={})\n  Category: {:?} → {:?}\n  Variables: {} → {}", 
                y_desc, expression, r_squared, fitness, complexity, x_category, y_category, x_desc, y_desc)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureCategory {
    PiecePosition,  // 0-767: Individual piece positions
    Strategic,      // 768-783: Core strategic evaluations
    Advanced,       // 784-855: Advanced chess features
    Derived,        // 856-1023: Mathematical transformations
    Unknown,        // Invalid index
}

impl Default for FeatureMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_descriptions() {
        let mapper = FeatureMapper::new();
        
        // Test piece position
        assert_eq!(mapper.describe_feature(0), "white_pawn_a1");
        assert_eq!(mapper.describe_feature(63), "white_pawn_h8");
        assert_eq!(mapper.describe_feature(64), "black_pawn_a1");
        
        // Test strategic features
        assert_eq!(mapper.describe_feature(768), "material_balance");
        assert_eq!(mapper.describe_feature(769), "positional_score");
        assert_eq!(mapper.describe_feature(770), "white_king_safety");
        
        // Test categories
        assert_eq!(mapper.get_feature_category(0), FeatureCategory::PiecePosition);
        assert_eq!(mapper.get_feature_category(768), FeatureCategory::Strategic);
        assert_eq!(mapper.get_feature_category(800), FeatureCategory::Advanced);
        assert_eq!(mapper.get_feature_category(900), FeatureCategory::Derived);
    }
    
    #[test]
    fn test_piece_interpretation() {
        let mapper = FeatureMapper::new();
        
        // Test square extraction
        assert_eq!(mapper.get_square_from_piece_index(0), Some("a1".to_string()));
        assert_eq!(mapper.get_square_from_piece_index(63), Some("h8".to_string()));
        
        // Test piece type extraction
        assert_eq!(mapper.get_piece_from_index(0), Some((PieceType::Pawn, Color::White)));
        assert_eq!(mapper.get_piece_from_index(64), Some((PieceType::Pawn, Color::Black)));
        assert_eq!(mapper.get_piece_from_index(128), Some((PieceType::Knight, Color::White)));
    }
}