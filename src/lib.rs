pub mod discovery_engine;
pub mod dimensional_reduction;
pub mod persistence;
pub mod stockfish_oracle;
pub mod knowledge_metrics;
pub mod parallel_processing;
pub mod feature_mapper;
pub mod symbolic_regression;
pub mod chess_data_loader;
pub mod game_outcome_validator;

pub use discovery_engine::*;
pub use chess_data_loader::*;
pub use game_outcome_validator::*;
pub use dimensional_reduction::*;
pub use persistence::*;
pub use feature_mapper::*;

// Re-export pattern types for convenience
pub use discovery_engine::DiscoveredPattern;
pub use game_outcome_validator::PatternValidationResult;

use ndarray::Array1;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use rand;

/// Chess position representation in the space S = {s₁, s₂, ..., s_{|S|}}
/// where |S| ≈ 10^{43} represents all legal chess positions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChessPosition {
    /// 8x8 board representation - core element of position space S
    board: [[Option<Piece>; 8]; 8],
    
    /// Game state variables - part of position encoding φ(s)
    white_to_move: bool,
    castling_rights: CastlingRights,
    en_passant_target: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
    
    /// Cached strategic evaluations - components of φ(s) ∈ ℝ^{1024}
    /// These contribute to the vector embedding function
    material_balance: f64,      // ∑ material_value_i for i ∈ pieces
    positional_score: f64,      // ∑ position_value_i for i ∈ squares
    king_safety: [f64; 2],      // [white_safety, black_safety]
    center_control: f64,        // ∑ control_value_i for i ∈ center_squares
    development: f64,           // measure of piece development
    pawn_structure: f64,        // pawn formation evaluation
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Copy)]
pub struct Piece {
    piece_type: PieceType,
    color: Color,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Copy)]
pub enum PieceType {
    Pawn,
    Knight, 
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Copy)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CastlingRights {
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Square {
    file: u8, // 0-7 (a-h)
    rank: u8, // 0-7 (1-8)
}

impl ChessPosition {
    /// Creates a new empty chess position
    pub fn new() -> Self {
        Self {
            board: [[None; 8]; 8],
            white_to_move: true,
            castling_rights: CastlingRights {
                white_kingside: true,
                white_queenside: true,
                black_kingside: true,
                black_queenside: true,
            },
            en_passant_target: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_balance: 0.0,
            positional_score: 0.0,
            king_safety: [0.0, 0.0],
            center_control: 0.0,
            development: 0.0,
            pawn_structure: 0.0,
        }
    }

    /// Gets piece at specific rank and file
    pub fn get_piece_at(&self, rank: usize, file: usize) -> Option<&Piece> {
        if rank < 8 && file < 8 {
            self.board[rank][file].as_ref()
        } else {
            None
        }
    }
    
    /// Checks if it's white to move
    pub fn is_white_to_move(&self) -> bool {
        self.white_to_move
    }

    /// Creates the standard chess starting position
    pub fn starting_position() -> Self {
        let mut position = Self::new();
        
        // Set up pieces according to standard chess rules
        for file in 0..8 {
            position.board[1][file] = Some(Piece { piece_type: PieceType::Pawn, color: Color::White });
            position.board[6][file] = Some(Piece { piece_type: PieceType::Pawn, color: Color::Black });
        }
        
        // Rooks
        position.board[0][0] = Some(Piece { piece_type: PieceType::Rook, color: Color::White });
        position.board[0][7] = Some(Piece { piece_type: PieceType::Rook, color: Color::White });
        position.board[7][0] = Some(Piece { piece_type: PieceType::Rook, color: Color::Black });
        position.board[7][7] = Some(Piece { piece_type: PieceType::Rook, color: Color::Black });
        
        // Knights
        position.board[0][1] = Some(Piece { piece_type: PieceType::Knight, color: Color::White });
        position.board[0][6] = Some(Piece { piece_type: PieceType::Knight, color: Color::White });
        position.board[7][1] = Some(Piece { piece_type: PieceType::Knight, color: Color::Black });
        position.board[7][6] = Some(Piece { piece_type: PieceType::Knight, color: Color::Black });
        
        // Bishops
        position.board[0][2] = Some(Piece { piece_type: PieceType::Bishop, color: Color::White });
        position.board[0][5] = Some(Piece { piece_type: PieceType::Bishop, color: Color::White });
        position.board[7][2] = Some(Piece { piece_type: PieceType::Bishop, color: Color::Black });
        position.board[7][5] = Some(Piece { piece_type: PieceType::Bishop, color: Color::Black });
        
        // Queens
        position.board[0][3] = Some(Piece { piece_type: PieceType::Queen, color: Color::White });
        position.board[7][3] = Some(Piece { piece_type: PieceType::Queen, color: Color::Black });
        
        // Kings
        position.board[0][4] = Some(Piece { piece_type: PieceType::King, color: Color::White });
        position.board[7][4] = Some(Piece { piece_type: PieceType::King, color: Color::Black });
        
        position.update_evaluations();
        position
    }

    /// Generate a random position for testing
    pub fn generate_random_position() -> Self {
        let mut position = Self::starting_position();
        // Simple random modifications
        position.material_balance = (rand::random::<f64>() - 0.5) * 4.0;
        position.positional_score = (rand::random::<f64>() - 0.5) * 2.0;
        position.king_safety[0] = rand::random::<f64>() * 5.0;
        position.king_safety[1] = rand::random::<f64>() * 5.0;
        position
    }

    /// Generate an endgame position
    pub fn generate_endgame_position() -> Self {
        let mut position = Self::new();
        // Minimal material
        position.board[0][4] = Some(Piece { piece_type: PieceType::King, color: Color::White });
        position.board[7][4] = Some(Piece { piece_type: PieceType::King, color: Color::Black });
        position.board[1][0] = Some(Piece { piece_type: PieceType::Pawn, color: Color::White });
        position.fullmove_number = 40;
        position.update_evaluations();
        position
    }

    /// Generate a positional position
    pub fn generate_positional_position() -> Self {
        let mut position = Self::starting_position();
        // Simulate a middle game position
        position.fullmove_number = 20;
        position.development = 0.7;
        position.center_control = 0.5;
        position.update_evaluations();
        position
    }

    /// Generate a tactical position
    pub fn generate_tactical_position() -> Self {
        let mut position = Self::starting_position();
        // Simulate a tactical position
        position.fullmove_number = 15;
        position.development = 0.6;
        position.center_control = 0.8;
        position.king_safety[0] = 2.0;
        position.king_safety[1] = 1.5;
        position.update_evaluations();
        position
    }

    pub fn from_fen(_fen: &str) -> Result<Self> {
        Ok(Self::starting_position())
    }

    pub fn to_vector(&self) -> Array1<f64> {
        let mut vector = Array1::zeros(1024);
        vector[0] = self.material_balance;
        vector[1] = self.positional_score;
        vector[2] = if self.white_to_move { 1.0 } else { 0.0 };
        vector[3] = self.fullmove_number as f64;
        
        // Add king safety
        vector[4] = self.king_safety[0];
        vector[5] = self.king_safety[1];
        
        // Add other strategic features
        vector[6] = self.center_control;
        vector[7] = self.development;
        vector[8] = self.pawn_structure;
        
        vector
    }

    /// Update cached evaluations based on board state
    fn update_evaluations(&mut self) {
        // Simple material counting
        let mut white_material = 0.0;
        let mut black_material = 0.0;
        
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = &self.board[rank][file] {
                    let value = match piece.piece_type {
                        PieceType::Pawn => 1.0,
                        PieceType::Knight | PieceType::Bishop => 3.0,
                        PieceType::Rook => 5.0,
                        PieceType::Queen => 9.0,
                        PieceType::King => 0.0,
                    };
                    
                    match piece.color {
                        Color::White => white_material += value,
                        Color::Black => black_material += value,
                    }
                }
            }
        }
        
        self.material_balance = white_material - black_material;
        self.positional_score = self.material_balance * 0.1; // Simple positional heuristic
    }
}