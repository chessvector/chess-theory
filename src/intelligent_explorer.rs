use anyhow::Result;
use crate::chess_data_loader::ChessDataLoader;
use crate::ChessPosition;
use std::collections::{HashMap, HashSet};

/// Well-studied chess openings to start exploration from
#[derive(Debug, Clone)]
pub struct OpeningFamily {
    pub name: String,
    pub eco_code: String,
    pub base_position: ChessPosition,
    pub key_moves: Vec<String>,
    pub popularity: f64,  // 0.0 to 1.0
}

/// Tracks discovery progress and guides intelligent pivoting
#[derive(Debug, Clone)]
pub struct DiscoveryState {
    pub current_opening: String,
    pub positions_explored: usize,
    pub constants_found_recently: usize,
    pub cycles_without_constants: usize,
    pub total_constants_found: usize,
    pub explored_openings: HashSet<String>,
    pub promising_branches: Vec<String>,
}

/// Intelligent position explorer that adapts based on discoveries
pub struct IntelligentExplorer {
    pub loader: ChessDataLoader,
    pub state: DiscoveryState,
    pub opening_families: Vec<OpeningFamily>,
    pub position_cache: HashMap<String, Vec<ChessPosition>>,
    pub exploration_depth: usize,
    pub pivot_threshold: usize,  // Cycles without constants before pivoting
}

impl IntelligentExplorer {
    pub fn new() -> Result<Self> {
        let mut explorer = Self {
            loader: ChessDataLoader::new(),
            state: DiscoveryState {
                current_opening: "Italian Game".to_string(),
                positions_explored: 0,
                constants_found_recently: 0,
                cycles_without_constants: 0,
                total_constants_found: 0,
                explored_openings: HashSet::new(),
                promising_branches: Vec::new(),
            },
            opening_families: Self::initialize_opening_families(),
            position_cache: HashMap::new(),
            exploration_depth: 10,  // How deep to explore variations
            pivot_threshold: 3,     // Pivot after 3 cycles without constants
        };
        
        Ok(explorer)
    }
    
    /// Initialize well-studied opening families
    fn initialize_opening_families() -> Vec<OpeningFamily> {
        vec![
            // Most popular e4 openings
            OpeningFamily {
                name: "Italian Game".to_string(),
                eco_code: "C50-C59".to_string(),
                base_position: Self::create_position_after_moves(&["e4", "e5", "Nf3", "Nc6", "Bc4"]),
                key_moves: vec!["e4".to_string(), "e5".to_string(), "Nf3".to_string(), "Nc6".to_string(), "Bc4".to_string()],
                popularity: 0.95,
            },
            OpeningFamily {
                name: "Spanish Opening (Ruy Lopez)".to_string(),
                eco_code: "C60-C99".to_string(),
                base_position: Self::create_position_after_moves(&["e4", "e5", "Nf3", "Nc6", "Bb5"]),
                key_moves: vec!["e4".to_string(), "e5".to_string(), "Nf3".to_string(), "Nc6".to_string(), "Bb5".to_string()],
                popularity: 0.98,
            },
            OpeningFamily {
                name: "Sicilian Defense".to_string(),
                eco_code: "B20-B99".to_string(),
                base_position: Self::create_position_after_moves(&["e4", "c5"]),
                key_moves: vec!["e4".to_string(), "c5".to_string()],
                popularity: 1.0,
            },
            OpeningFamily {
                name: "French Defense".to_string(),
                eco_code: "C00-C19".to_string(),
                base_position: Self::create_position_after_moves(&["e4", "e6"]),
                key_moves: vec!["e4".to_string(), "e6".to_string()],
                popularity: 0.85,
            },
            OpeningFamily {
                name: "Caro-Kann Defense".to_string(),
                eco_code: "B10-B19".to_string(),
                base_position: Self::create_position_after_moves(&["e4", "c6"]),
                key_moves: vec!["e4".to_string(), "c6".to_string()],
                popularity: 0.80,
            },
            
            // Most popular d4 openings
            OpeningFamily {
                name: "Queen's Gambit".to_string(),
                eco_code: "D06-D69".to_string(),
                base_position: Self::create_position_after_moves(&["d4", "d5", "c4"]),
                key_moves: vec!["d4".to_string(), "d5".to_string(), "c4".to_string()],
                popularity: 0.90,
            },
            OpeningFamily {
                name: "King's Indian Defense".to_string(),
                eco_code: "E60-E99".to_string(),
                base_position: Self::create_position_after_moves(&["d4", "Nf6", "c4", "g6"]),
                key_moves: vec!["d4".to_string(), "Nf6".to_string(), "c4".to_string(), "g6".to_string()],
                popularity: 0.88,
            },
            OpeningFamily {
                name: "Nimzo-Indian Defense".to_string(),
                eco_code: "E20-E59".to_string(),
                base_position: Self::create_position_after_moves(&["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]),
                key_moves: vec!["d4".to_string(), "Nf6".to_string(), "c4".to_string(), "e6".to_string(), "Nc3".to_string(), "Bb4".to_string()],
                popularity: 0.92,
            },
            
            // Other popular openings
            OpeningFamily {
                name: "English Opening".to_string(),
                eco_code: "A10-A39".to_string(),
                base_position: Self::create_position_after_moves(&["c4"]),
                key_moves: vec!["c4".to_string()],
                popularity: 0.75,
            },
            OpeningFamily {
                name: "Réti Opening".to_string(),
                eco_code: "A04-A09".to_string(),
                base_position: Self::create_position_after_moves(&["Nf3"]),
                key_moves: vec!["Nf3".to_string()],
                popularity: 0.70,
            },
        ]
    }
    
    /// Create a position after a sequence of moves (simplified)
    fn create_position_after_moves(moves: &[&str]) -> ChessPosition {
        let mut pos = ChessPosition::starting_position();
        pos.fullmove_number = moves.len() as u32 / 2 + 1;
        // In real implementation, would apply moves to update position
        pos
    }
    
    /// Generate positions for the next discovery cycle
    pub fn generate_next_batch(&mut self, batch_size: usize) -> Result<Vec<ChessPosition>> {
        // Check if we need to pivot to a new opening
        if self.should_pivot() {
            self.pivot_to_new_opening()?;
        }
        
        let mut positions = Vec::new();
        
        // Get current opening family
        let current_family = self.opening_families.iter()
            .find(|f| f.name == self.state.current_opening)
            .cloned()
            .unwrap_or_else(|| self.opening_families[0].clone());
        
        // Start with the base position
        positions.push(current_family.base_position.clone());
        
        // Generate variations from this opening
        positions.extend(self.generate_opening_variations(&current_family, batch_size - 1)?);
        
        // Update state
        self.state.positions_explored += positions.len();
        
        println!("🎯 Exploring {} positions from {} (ECO: {})", 
                 positions.len(), current_family.name, current_family.eco_code);
        println!("   Total positions explored: {}", self.state.positions_explored);
        println!("   Cycles without constants: {}", self.state.cycles_without_constants);
        
        Ok(positions)
    }
    
    /// Generate variations from a specific opening
    fn generate_opening_variations(&mut self, opening: &OpeningFamily, count: usize) -> Result<Vec<ChessPosition>> {
        let mut variations = Vec::new();
        let base = opening.base_position.clone();
        
        // Generate main line continuations with STRATEGIC DIVERSITY
        for depth in 1..=self.exploration_depth {
            if variations.len() >= count {
                break;
            }
            
            let mut variation = base.clone();
            variation.set_fullmove_number(base.fullmove_number + depth as u32);
            
            // Create DIVERSE strategic scenarios for constants discovery
            let scenario = depth % 8; // 8 different strategic scenarios
            
            let (material_balance, positional_score, king_safety, center_control, development, pawn_structure) = match scenario {
                0 => {
                    // Material advantage scenario
                    let material = 1.5 + (depth as f64 * 0.2);  // White advantage
                    let positional = -0.3 + (depth as f64 * 0.1); // Positional compensation
                    let king_safety = if opening.name.contains("Sicilian") { [2.8, 4.2] } else { [3.5, 3.8] };
                    (material, positional, king_safety, 3.0 + (depth as f64 * 0.1), depth as f64 * 1.5, 0.8)
                },
                1 => {
                    // Material disadvantage scenario
                    let material = -1.2 - (depth as f64 * 0.15); // Black advantage
                    let positional = 1.8 + (depth as f64 * 0.2);  // White positional compensation
                    let king_safety = if opening.name.contains("Defense") { [4.5, 2.5] } else { [4.0, 3.0] };
                    (material, positional, king_safety, 5.5 + (depth as f64 * 0.2), depth as f64 * 2.2, 1.2)
                },
                2 => {
                    // King safety imbalance scenario
                    let material = 0.0 + (depth as f64 * 0.05); // Balanced material
                    let positional = 0.5 + (depth as f64 * 0.1);
                    let king_safety = [1.5 - (depth as f64 * 0.1), 5.5 + (depth as f64 * 0.1)]; // White king exposed
                    (material, positional, king_safety, 4.0, depth as f64 * 1.8, 0.5)
                },
                3 => {
                    // Development advantage scenario
                    let material = -0.2; // Slight material deficit
                    let positional = 1.5 + (depth as f64 * 0.3); // But strong development
                    let king_safety = [4.2, 3.8];
                    let development = 8.0 + (depth as f64 * 2.0); // Strong development lead
                    (material, positional, king_safety, 6.0 + (depth as f64 * 0.3), development, 1.5)
                },
                4 => {
                    // Center control focus scenario
                    let material = 0.3;
                    let positional = 0.8 + (depth as f64 * 0.2);
                    let king_safety = [3.8, 4.0];
                    let center_control = 7.5 + (depth as f64 * 0.4); // Dominant center
                    (material, positional, king_safety, center_control, depth as f64 * 1.2, 2.0)
                },
                5 => {
                    // Pawn structure advantage scenario
                    let material = -0.5; // Material deficit
                    let positional = 2.2 + (depth as f64 * 0.25); // Superior structure
                    let king_safety = [4.8, 3.2];
                    let pawn_structure = 3.5 + (depth as f64 * 0.3); // Excellent pawn structure
                    (material, positional, king_safety, 4.5, depth as f64 * 1.0, pawn_structure)
                },
                6 => {
                    // Tactical complications scenario
                    let material = 2.8 + (depth as f64 * 0.4); // Sharp material imbalance
                    let positional = -1.5 - (depth as f64 * 0.2); // Risky position
                    let king_safety = [2.0, 1.8]; // Both kings exposed
                    (material, positional, king_safety, 3.0, depth as f64 * 3.0, -0.5)
                },
                _ => {
                    // Balanced but complex scenario
                    let material = 0.1 * ((depth as f64).sin()); // Oscillating balance
                    let positional = 1.0 + (depth as f64 * 0.15);
                    let king_safety = [3.5 + (depth as f64 * 0.05), 3.5 - (depth as f64 * 0.05)];
                    (material, positional, king_safety, 4.5, depth as f64 * 1.6, 1.0)
                }
            };
            
            variation.set_strategic_evaluations(
                material_balance,
                positional_score,
                king_safety,
                center_control,
                development,
                pawn_structure
            );
            
            variations.push(variation);
        }
        
        // Add DIVERSE sideline variations for comprehensive strategic exploration
        for i in 0..count.saturating_sub(variations.len()) {
            let mut sideline = base.clone();
            sideline.fullmove_number = base.fullmove_number + (i as u32 % 10) + 1;
            
            // Create EXTREME strategic imbalances to test constants across wide ranges
            let variety = i % 12; // 12 different strategic varieties
            
            let (material, positional, king_safety, center, development, pawn_structure) = match variety {
                0 => {
                    // EXTREME material advantage
                    let mat = 3.5 + (i as f64 * 0.1);
                    (mat, -1.0, [1.8, 4.8], 2.5, i as f64 * 0.8, 0.2)
                },
                1 => {
                    // EXTREME material disadvantage  
                    let mat = -3.2 - (i as f64 * 0.12);
                    (mat, 2.8, [5.2, 1.5], 7.0, i as f64 * 2.5, 2.8)
                },
                2 => {
                    // CRITICAL king safety crisis
                    (0.8, 1.2, [0.5, 0.8], 3.5, i as f64 * 1.2, 1.0)
                },
                3 => {
                    // FORTRESS-like king safety
                    (-0.8, 0.5, [6.0, 5.8], 2.0, i as f64 * 0.5, 1.8)
                },
                4 => {
                    // HYPERMODERN center strategy
                    (0.2, 1.8, [4.2, 3.8], 8.5 + (i as f64 * 0.2), i as f64 * 1.8, 2.5)
                },
                5 => {
                    // CLASSICAL center dominance
                    (1.2, 0.8, [3.2, 4.0], 9.2 + (i as f64 * 0.3), i as f64 * 2.0, 1.2)
                },
                6 => {
                    // RAPID development sacrifice
                    (-1.5, 3.2, [3.0, 4.5], 6.5, 12.0 + (i as f64 * 1.5), 0.8)
                },
                7 => {
                    // SLOW positional buildup
                    (0.3, 2.5, [4.8, 4.2], 5.5, 2.0 + (i as f64 * 0.3), 3.2)
                },
                8 => {
                    // Sicilian-specific: SHARP tactical positions
                    let sharpness = if opening.name.contains("Sicilian") { 2.0 } else { 1.0 };
                    (1.8 * sharpness, -0.8, [2.2, 2.5], 4.0, i as f64 * 2.8, -0.5)
                },
                9 => {
                    // Defense-specific: SOLID counterplay
                    let solidity = if opening.name.contains("Defense") { 1.8 } else { 1.0 };
                    (-0.5, 2.5 * solidity, [4.8, 3.2], 6.8, i as f64 * 1.2, 2.8)
                },
                10 => {
                    // COMPLEX sacrificial positions
                    (4.2 - (i as f64 * 0.3), -2.0 + (i as f64 * 0.2), [1.2, 3.8], 3.5, i as f64 * 4.0, -1.2)
                },
                _ => {
                    // ENDGAME-like simplification
                    let endgame_factor = 0.3 + (i as f64 * 0.05);
                    (endgame_factor, 1.8, [4.5, 4.5], 3.0, i as f64 * 0.8, 2.5)
                }
            };
            
            sideline.set_strategic_evaluations(
                material,
                positional, 
                king_safety,
                center,
                development,
                pawn_structure
            );
            
            variations.push(sideline);
        }
        
        Ok(variations)
    }
    
    /// Determine if we should pivot to a new opening
    fn should_pivot(&self) -> bool {
        self.state.cycles_without_constants >= self.pivot_threshold
    }
    
    /// Pivot to a new opening family
    fn pivot_to_new_opening(&mut self) -> Result<()> {
        // Mark current opening as explored
        self.state.explored_openings.insert(self.state.current_opening.clone());
        
        // Find next unexplored opening with highest popularity
        let next_opening = self.opening_families.iter()
            .filter(|f| !self.state.explored_openings.contains(&f.name))
            .max_by(|a, b| a.popularity.partial_cmp(&b.popularity).unwrap())
            .cloned();
        
        match next_opening {
            Some(opening) => {
                println!("🔄 Pivoting from {} to {} (popularity: {:.2})", 
                         self.state.current_opening, opening.name, opening.popularity);
                self.state.current_opening = opening.name;
                self.state.cycles_without_constants = 0;
            }
            None => {
                // All openings explored, start second pass with deeper analysis
                println!("🔄 All openings explored. Starting deeper analysis...");
                self.state.explored_openings.clear();
                self.exploration_depth += 5;
                self.state.current_opening = "Sicilian Defense".to_string(); // Start with most popular
            }
        }
        
        Ok(())
    }
    
    /// Update discovery state based on results
    pub fn update_state(&mut self, constants_found: usize) {
        self.state.constants_found_recently = constants_found;
        self.state.total_constants_found += constants_found;
        
        if constants_found == 0 {
            self.state.cycles_without_constants += 1;
        } else {
            self.state.cycles_without_constants = 0;
            // Mark this branch as promising
            self.state.promising_branches.push(self.state.current_opening.clone());
        }
        
        println!("📊 Discovery state updated:");
        println!("   Constants found this cycle: {}", constants_found);
        println!("   Total constants found: {}", self.state.total_constants_found);
        println!("   Promising branches: {:?}", self.state.promising_branches);
    }
    
    /// Get discovery progress report
    pub fn get_progress_report(&self) -> String {
        format!(
            "🎯 Intelligent Explorer Progress:\n\
             Current Opening: {}\n\
             Positions Explored: {}\n\
             Total Constants Found: {}\n\
             Openings Analyzed: {}/{}\n\
             Promising Branches: {:?}\n\
             Exploration Depth: {}",
            self.state.current_opening,
            self.state.positions_explored,
            self.state.total_constants_found,
            self.state.explored_openings.len(),
            self.opening_families.len(),
            self.state.promising_branches,
            self.exploration_depth
        )
    }
}