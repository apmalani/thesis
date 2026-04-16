"""
Redistricting Environment with Graph-Based State Representation.

This module implements a Gymnasium environment for redistricting optimization
that uses graph-based state representations suitable for GNN encoding.
"""

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*torch_geometric.distributed.*")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.construction import build_precinct_graph
from graph.metrics import MCalc
from analysis.reward_function import GerrymanderingRewardFunction


class GerrymanderingEnv(gym.Env):
    """
    Redistricting environment with graph-based state representation.
    
    The state is represented as a graph where:
    - Nodes are precincts with demographic/partisan features
    - Edges represent adjacency relationships
    
    The action space is fixed at initialization based on the initial number
    of valid precinct reassignments (e.g., 1068 for Arizona). This ensures
    compatibility with the agent's policy network, which expects a fixed
    action dimension.
    - Nodes are precincts with demographic/partisan features
    - Edges represent adjacency between precincts
    """
    
    def __init__(
        self,
        state: str,
        basepath: str,
        pop_tol: float = 0.05,
        reward_weights: Optional[Dict] = None,
        max_steps: int = 1000,
        skip_compactness: bool = False
    ):
        """
        Initialize redistricting environment.
        
        Args:
            state: State abbreviation (e.g., 'az')
            basepath: Base path to processed data
            pop_tol: Population tolerance for districts
            reward_weights: Weights for reward components
            max_steps: Maximum steps per episode
            skip_compactness: If True, skip compactness (Polsby-Popper) in reward
        """
        super(GerrymanderingEnv, self).__init__()
        
        self.state = state
        self.basepath = basepath
        self.pop_tol = pop_tol
        self.max_steps = max_steps
        self.current_step = 0
        self.skip_compactness = skip_compactness
        
        # Build precinct graph and initial partition
        self.graph, self.partition = build_precinct_graph(state, basepath)
        self.metrics_calc = MCalc()
        
        # Store baseline assignment for exploration bonus calculation
        self._baseline_assignment = dict(self.partition.assignment)
        
        # Track previous score for delta reward calculation
        self._previous_score = None
        
        # Pre-calculate geometry data (Area, Perimeter) for compactness checks
        # This avoids repeated geometry calculations during training
        self._precinct_geometry_cache = {}
        for node in self.graph.nodes():
            if 'geometry' in self.graph.nodes[node]:
                geom = self.graph.nodes[node]['geometry']
                self._precinct_geometry_cache[node] = {
                    'area': geom.area,
                    'perimeter': geom.length
                }
        
        # Construct baseline stats path (handle both cases: basepath with/without state)
        from pathlib import Path
        base_path = Path(basepath)
        if base_path.name == state:
            # basepath already includes state
            baseline_path = base_path / "baseline_stats.csv"
        else:
            # basepath is parent directory
            baseline_path = base_path / state / "baseline_stats.csv"
        
        self.reward_fn = GerrymanderingRewardFunction(str(baseline_path))
        
        # Default reward weights (Multi-objective with hard constraints)
        if reward_weights is None:
            self.reward_weights = {
                'PopPenalty': 10.0,  # High weight for hard constraint
                'EfficiencyGap': 2.0,
                'Compactness': 1.0,
                'Minority': 1.0
            }
        else:
            self.reward_weights = reward_weights
        
        self.n_precincts = len(self.graph.nodes)
        self.n_districts = len(self.partition.parts)
        
        # Initialize caches for performance optimization
        self._node_feature_totals = None
        self._cached_graph_observation = None
        self._partition_hash = None
        self._modified_districts = set()
        self._last_action_node = None
        
        # Generate valid actions (precinct reassignments)
        # Store initial action space size - this must remain fixed for agent compatibility
        self._valid_actions = self._generate_valid_actions()
        self._initial_action_space_size = len(self._valid_actions)
        self.action_space = spaces.Discrete(self._initial_action_space_size)
        
        # Observation space is now a graph (handled separately)
        # We keep a dummy observation space for Gymnasium compatibility
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(1,),  # Dummy shape
            dtype=np.float32
        )
    
    def _generate_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Generate list of valid actions (precinct reassignments).
        
        HARD ACTION MASKING: Only includes actions that maintain:
        1. Population constraints: Districts stay within min/max allowed population
        2. Contiguity: Districts remain connected after move
        
        For every move (Precinct P from District A to District B):
        a) Check Population: If (Dist_B_Pop + P_Pop) > Max_Allowed or (Dist_A_Pop - P_Pop) < Min_Allowed, mark as INVALID
        b) Check Contiguity: If removing P from Dist A makes Dist A non-contiguous, mark as INVALID
        
        Returns:
            List of (node, neighbor) tuples representing valid moves
        """
        # Calculate population bounds
        total_pop = sum(self.graph.nodes[n]['P0010001'] for n in self.graph.nodes())
        ideal_pop = total_pop / self.n_districts
        max_allowed_pop = ideal_pop * (1.0 + self.pop_tol)
        min_allowed_pop = ideal_pop * (1.0 - self.pop_tol)
        
        valid_actions = []
        for node in self.graph.nodes():
            current_district = self.partition.assignment[node]
            node_pop = self.graph.nodes[node]['P0010001']
            
            for neighbor in self.graph.neighbors(node):
                target_district = self.partition.assignment[neighbor]
                
                # Skip if same district
                if current_district == target_district:
                    continue
                
                # Create temporary assignment to test the move
                temp_assignment = dict(self.partition.assignment)
                temp_assignment[node] = target_district
                
                # Check 1: POPULATION CONSTRAINT
                # Calculate populations after move
                source_dist_pop_after = sum(
                    self.graph.nodes[n]['P0010001']
                    for n in self.graph.nodes()
                    if temp_assignment[n] == current_district
                )
                target_dist_pop_after = sum(
                    self.graph.nodes[n]['P0010001']
                    for n in self.graph.nodes()
                    if temp_assignment[n] == target_district
                )
                
                # Check if move violates population constraints
                if source_dist_pop_after < min_allowed_pop:
                    continue  # Source district would be too small
                if target_dist_pop_after > max_allowed_pop:
                    continue  # Target district would be too large
                
                # Check 2: CONTIGUITY CONSTRAINT
                # Verify source district remains contiguous after removal
                if not self._check_contiguity(temp_assignment, current_district):
                    continue  # Skip - would break contiguity
                
                # Verify target district remains contiguous after addition
                if not self._check_contiguity(temp_assignment, target_district):
                    continue  # Skip - would break contiguity
                
                # Action is valid - add it
                valid_actions.append((node, neighbor))
        
        return valid_actions
    
    def _update_valid_actions_incremental(
        self, 
        old_district: int, 
        new_district: int, 
        moved_node: int
    ) -> List[Tuple[int, int]]:
        """
        Incrementally update valid actions by only checking modified districts.
        
        This is much faster than regenerating all actions from scratch.
        Only re-checks nodes in affected districts and their neighbors.
        
        Args:
            old_district: District the node was moved from
            new_district: District the node was moved to
            moved_node: Node that was moved
            
        Returns:
            Updated list of valid actions
        """
        # Start with existing valid actions, remove invalidated ones
        valid_actions = [
            (n, nb) for n, nb in self._valid_actions
            if n != moved_node  # Remove actions involving moved node
        ]
        
        # Add new valid actions from modified districts
        # Only check nodes in affected districts and their neighbors
        affected_nodes = set()
        
        # Add nodes from old district
        for node in self.graph.nodes():
            if self.partition.assignment[node] == old_district:
                affected_nodes.add(node)
                affected_nodes.update(self.graph.neighbors(node))
        
        # Add nodes from new district
        for node in self.graph.nodes():
            if self.partition.assignment[node] == new_district:
                affected_nodes.add(node)
                affected_nodes.update(self.graph.neighbors(node))
        
        # Check new valid actions only in affected neighborhood
        # Enforce hard masking: verify each action maintains population and contiguity
        # Calculate population bounds
        total_pop = sum(self.graph.nodes[n]['P0010001'] for n in self.graph.nodes())
        ideal_pop = total_pop / self.n_districts
        max_allowed_pop = ideal_pop * (1.0 + self.pop_tol)
        min_allowed_pop = ideal_pop * (1.0 - self.pop_tol)
        
        for node in affected_nodes:
            current_district = self.partition.assignment[node]
            node_pop = self.graph.nodes[node]['P0010001']
            
            for neighbor in self.graph.neighbors(node):
                target_district = self.partition.assignment[neighbor]
                
                # Skip if same district
                if current_district == target_district:
                    continue
                
                action = (node, neighbor)
                if action in valid_actions:
                    continue  # Already validated
                
                # Create temporary assignment to test the move
                temp_assignment = dict(self.partition.assignment)
                temp_assignment[node] = target_district
                
                # Check 1: POPULATION CONSTRAINT
                source_dist_pop_after = sum(
                    self.graph.nodes[n]['P0010001']
                    for n in self.graph.nodes()
                    if temp_assignment[n] == current_district
                )
                target_dist_pop_after = sum(
                    self.graph.nodes[n]['P0010001']
                    for n in self.graph.nodes()
                    if temp_assignment[n] == target_district
                )
                
                if source_dist_pop_after < min_allowed_pop:
                    continue  # Source district would be too small
                if target_dist_pop_after > max_allowed_pop:
                    continue  # Target district would be too large
                
                # Check 2: CONTIGUITY CONSTRAINT
                if not self._check_contiguity(temp_assignment, current_district):
                    continue  # Skip - would break contiguity
                
                if not self._check_contiguity(temp_assignment, target_district):
                    continue  # Skip - would break contiguity
                
                # Action is valid - add it
                valid_actions.append(action)
        
        return valid_actions
    
    def _extract_node_features(self) -> np.ndarray:
        """
        Extract node features for GNN encoding.
        
        Features include:
        - Demographic data (population, voting age population, race/ethnicity)
        - Partisan data (Democratic/Republican votes)
        - District assignment (one-hot encoded)
        - Geographic features (normalized)
        
        Returns:
            Node feature matrix [num_nodes, feature_dim]
        """
        node_features = []
        
        # Get total values for normalization (cached for performance)
        if self._node_feature_totals is None:
            self._node_feature_totals = {
                'pop': sum(self.graph.nodes[n].get('P0010001', 0) for n in self.graph.nodes()),
                'vap': sum(self.graph.nodes[n].get('P0040001', 0) for n in self.graph.nodes()),
                'dem': sum(self.graph.nodes[n].get('CompDemVot', 0) for n in self.graph.nodes()),
                'rep': sum(self.graph.nodes[n].get('CompRepVot', 0) for n in self.graph.nodes())
            }
            self._node_feature_totals['votes'] = (
                self._node_feature_totals['dem'] + self._node_feature_totals['rep']
                if (self._node_feature_totals['dem'] + self._node_feature_totals['rep']) > 0 else 1
            )
        
        total_pop = self._node_feature_totals['pop']
        total_vap = self._node_feature_totals['vap']
        total_dem = self._node_feature_totals['dem']
        total_rep = self._node_feature_totals['rep']
        total_votes = self._node_feature_totals['votes']
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Population features (normalized)
            pop = node_data.get('P0010001', 0)
            vap = node_data.get('P0040001', 0)
            pop_frac = pop / total_pop if total_pop > 0 else 0
            vap_frac = vap / total_vap if total_vap > 0 else 0
            
            # Partisan features
            dem_votes = node_data.get('CompDemVot', 0)
            rep_votes = node_data.get('CompRepVot', 0)
            dem_frac = dem_votes / total_votes if total_votes > 0 else 0
            rep_frac = rep_votes / total_votes if total_votes > 0 else 0
            vote_margin = abs(dem_votes - rep_votes) / (dem_votes + rep_votes) if (dem_votes + rep_votes) > 0 else 0
            
            # Demographic features (race/ethnicity percentages)
            pct_white = node_data.get('P0040005', 0) / vap if vap > 0 else 0
            pct_latino = node_data.get('P0040002', 0) / vap if vap > 0 else 0
            pct_black = node_data.get('P0040006', 0) / vap if vap > 0 else 0
            pct_native = node_data.get('P0040007', 0) / vap if vap > 0 else 0
            pct_asian = node_data.get('P0040008', 0) / vap if vap > 0 else 0
            pct_nhpi = node_data.get('P0040009', 0) / vap if vap > 0 else 0
            pct_minority = 1 - pct_white
            
            # District assignment (one-hot encoded)
            district_id = self.partition.assignment[node]
            district_onehot = np.zeros(self.n_districts)
            if isinstance(district_id, (int, np.integer)) and 0 <= district_id < self.n_districts:
                district_onehot[district_id] = 1.0
            
            # Combine all features
            features = np.array([
                pop_frac,
                vap_frac,
                dem_frac,
                rep_frac,
                vote_margin,
                pct_white,
                pct_latino,
                pct_black,
                pct_native,
                pct_asian,
                pct_nhpi,
                pct_minority
            ], dtype=np.float32)
            
            # Concatenate with district one-hot
            features = np.concatenate([features, district_onehot])
            
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)
    
    def get_graph_observation(self) -> Tuple[nx.Graph, np.ndarray]:
        """
        Get graph-based observation for GNN encoding.
        
        Returns:
            Tuple of (graph, node_features)
        """
        # Cache graph observation to avoid redundant computation
        # Only regenerate if partition changed
        current_partition_hash = hash(tuple(sorted(self.partition.assignment.items())))
        
        if (self._cached_graph_observation is None or 
            self._partition_hash != current_partition_hash):
            node_features = self._extract_node_features()
            self._cached_graph_observation = (self.graph, node_features)
            self._partition_hash = current_partition_hash
        
        return self._cached_graph_observation
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation (dummy for Gymnasium compatibility).
        
        Returns:
            Dummy observation array
        """
        # This is a dummy observation for Gymnasium compatibility
        # The actual observation is the graph, accessed via get_graph_observation()
        return np.array([0.0], dtype=np.float32)
    
    def get_valid_action_mask(self) -> np.ndarray:
        """
        Get binary mask for valid actions.
        
        HARD ACTION MASKING: Invalid moves (those that break population constraints or contiguity)
        are masked out (set to 0 probability). The agent can never take an illegal action.
        
        Returns:
            Binary mask [action_space.n] where 1 indicates valid action
        """
        # Ensure mask size matches the fixed action space size
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        
        # Mark all actions in _valid_actions as valid (they've already been validated for
        # population constraints and contiguity in _generate_valid_actions)
        for i in range(min(len(self._valid_actions), self.action_space.n)):
            mask[i] = 1.0
        
        # All other actions are invalid (remain 0)
        return mask
    
    def _is_valid_action(self, action: int) -> bool:
        """
        Check if action maintains district contiguity.
        
        This checks for the "cut-vertex" problem: removing a precinct from
        its current district must not break that district's internal contiguity.
        
        Args:
            action: Action index
            
        Returns:
            True if action is valid
        """
        # Check bounds for both action space and valid actions list
        if action >= self.action_space.n or action >= len(self._valid_actions):
            return False
        
        node, neighbor = self._valid_actions[action]
        current_district = self.partition.assignment[node]
        target_district = self.partition.assignment[neighbor]
        
        if current_district == target_district:
            return False
        
        # Create temporary assignment to test the move
        temp_assignment = dict(self.partition.assignment)
        temp_assignment[node] = target_district
        
        # Check 1: Source district (current_district) remains contiguous after removal
        # This is the "cut-vertex" check: removing node should not disconnect the district
        if not self._check_contiguity(temp_assignment, current_district):
            return False
        
        # Check 2: Target district remains contiguous after addition
        if not self._check_contiguity(temp_assignment, target_district):
            return False
        
        return True
    
    def _check_contiguity(self, assignment: Dict, district: int) -> bool:
        """
        Check if district remains contiguous after reassignment.
        
        Uses networkx.is_connected() on the district subgraph to verify connectivity.
        This handles the "cut-vertex" problem: verifies that removing a node
        from a district does not break the district's connectivity.
        
        Args:
            assignment: District assignment dictionary
            district: District ID to check
            
        Returns:
            True if district is contiguous (connected)
        """
        district_nodes = [node for node, dist in assignment.items() if dist == district]
        
        # Empty or single-node districts are trivially contiguous
        if len(district_nodes) <= 1:
            return True
        
        # Fast path: if only 2 nodes, check if they're neighbors
        if len(district_nodes) == 2:
            return district_nodes[1] in self.graph.neighbors(district_nodes[0])
        
        # Create subgraph of the district
        district_subgraph = self.graph.subgraph(district_nodes)
        
        # Use networkx.is_connected() to verify connectivity
        return nx.is_connected(district_subgraph)
    
    def _check_population_balance(self, assignment: Dict) -> bool:
        """
        Check if districts meet population balance requirements.
        
        Args:
            assignment: District assignment dictionary
            
        Returns:
            True if all districts are within population tolerance
        """
        total_pop = sum(self.graph.nodes[n]['P0010001'] for n in self.graph.nodes())
        ideal_pop = total_pop / self.n_districts
        
        for district in range(self.n_districts):
            district_pop = sum(
                self.graph.nodes[n]['P0010001']
                for n in self.graph.nodes()
                if assignment[n] == district
            )
            deviation = abs(district_pop - ideal_pop) / ideal_pop
            if deviation > self.pop_tol:
                return False
        return True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: Action index
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Initialize done flag
        done = False
        
        # Initialize info dict early
        info = {}
        
        # Validate action is within bounds of current valid actions
        if action >= len(self._valid_actions):
            # Action index out of bounds - return penalty
            return self._get_observation(), -10.0, True, {
                "error": "Action index out of bounds",
                "action": action,
                "valid_actions_count": len(self._valid_actions),
                "action_space_size": self.action_space.n
            }
        
        node, neighbor = self._valid_actions[action]
        old_district = self.partition.assignment[node]
        new_district = self.partition.assignment[neighbor]
        
        # Apply action: reassign precinct
        # NOTE: Removed old_metrics calculation - it was never used and very expensive
        new_assignment = dict(self.partition.assignment)
        new_assignment[node] = new_district
        
        from gerrychain import Partition
        self.partition = Partition(
            self.graph, new_assignment, self.partition.updaters
        )
        
        # Incrementally update valid actions (only check modified districts)
        # This is much faster than regenerating all actions
        self._valid_actions = self._update_valid_actions_incremental(
            old_district, new_district, node
        )
        
        # Track modified districts for next step
        self._modified_districts = {old_district, new_district}
        self._last_action_node = node
        
        # Invalidate cached graph observation (partition changed)
        self._cached_graph_observation = None
        self._partition_hash = None
        
        # Calculate population deviation (for hard constraint)
        total_pop = sum(self.graph.nodes[n]['P0010001'] for n in self.graph.nodes())
        ideal_pop = total_pop / self.n_districts
        pop_deviations = []
        for district_id in self.partition.parts.keys():
            district_pop = sum(
                self.graph.nodes[n]['P0010001']
                for n in self.partition.parts[district_id]
            )
            deviation = abs(district_pop - ideal_pop) / ideal_pop
            pop_deviations.append(deviation)
        max_pop_deviation = max(pop_deviations) * 100  # Convert to percentage
        
        # Calculate reward using multi-objective function with hard constraints
        efficiency_gap = 0.0
        try:
            metrics_df = self.metrics_calc.calculate_metrics(
                self.partition, include_geometry=not self.skip_compactness
            )
            metrics = metrics_df.iloc[0].to_dict()
            efficiency_gap = metrics.get('EfficiencyGap', 0.0)  # Extract for logging
            
            # Use multi-objective reward function
            # Note: partition and graph parameters removed - connectivity is now enforced via hard action masking
            current_score, connectivity_violation = self.reward_fn.calculate_reward(
                metrics, 
                self.reward_weights,
                max_pop_deviation=max_pop_deviation,
                skip_compactness=self.skip_compactness
            )
            
            # Track if map is disconnected or illegal for logging
            is_disconnected = connectivity_violation
            is_illegal = max_pop_deviation > 5.0
            
            # SOFTENED HARD WALL: Do NOT terminate episode on illegal/disconnected states
            # Keep the negative reward (pain signal) but allow agent to continue and learn recovery
            if connectivity_violation:
                info['connectivity_violation'] = True
                info['is_disconnected'] = True
                info['error'] = "District contiguity violated - negative reward but continuing"
                # Don't set done = True - allow agent to recover
            
            # If illegal, mark it but don't terminate
            if is_illegal:
                info['is_illegal'] = True
            
        except Exception as e:
            # If metrics calculation fails, check connectivity and legality
            is_disconnected = False
            is_illegal = max_pop_deviation > 5.0
            
            # Check connectivity first
            try:
                for district_id in self.partition.parts.keys():
                    district_nodes = list(self.partition.parts[district_id])
                    if len(district_nodes) <= 1:
                        continue
                    district_subgraph = self.graph.subgraph(district_nodes)
                    if not nx.is_connected(district_subgraph):
                        is_disconnected = True
                        current_score = -100.0  # Keep negative reward but don't terminate
                        info['connectivity_violation'] = True
                        info['is_disconnected'] = True
                        info['error'] = "District contiguity violated - negative reward but continuing"
                        break
            except:
                pass  # If connectivity check fails, continue
            
            # If illegal (but not disconnected), also set to -100 but don't terminate
            if is_illegal and not is_disconnected:
                current_score = -100.0
                info['is_illegal'] = True
            elif not is_disconnected:
                # Legal and connected, but metrics failed - use 0 as fallback
                current_score = 0.0
                info['is_illegal'] = False
                info['is_disconnected'] = False
        
        # Calculate distance from baseline (for logging, regardless of legal/illegal state)
        distance_from_baseline = sum(
            1 for node in self.graph.nodes()
            if self.partition.assignment[node] != self._baseline_assignment.get(node, -1)
        )
        
        # SIMPLIFIED REWARD: Since illegal moves are now impossible (hard action masking),
        # reward is strictly the improvement in map quality (delta of total score)
        # No penalties needed - illegal actions are prevented, not penalized
        
        # DELTA REWARD: Reward is STRICTLY the DELTA (Improvement) of Total Score
        # Reward = (Current_Total_Score - Previous_Total_Score)
        # Use float64 for precision to prevent rounding to zero
        if self._previous_score is not None:
            delta = np.float64(current_score) - np.float64(self._previous_score)
        else:
            # First step: no previous score, use 0 as baseline
            delta = np.float64(0.0)
        
        # EXPLORATION BONUS: Reward distance from baseline map
        # Small exploration bonus: 0.01 per 100 precinct-swaps away from baseline
        # This encourages agent to explore beyond the starting map
        exploration_bonus = (distance_from_baseline / 100.0) * 0.01
        delta += exploration_bonus
        
        # REWARD SCALING: Multiply delta by 100 to make small score improvements visible
        # If 'Score' improvements are small (e.g., 0.01), the agent can't 'see' them
        # Multiplying by 100 makes 0.01 -> 1.0, which is large enough for gradients
        SCALE_FACTOR = 100.0
        reward = delta * SCALE_FACTOR
        
        # Convert back to float32 for neural network (after all calculations)
        reward = np.float32(reward)
        
        # Store distance and bonus for logging
        info['distance_from_baseline'] = distance_from_baseline
        info['exploration_bonus'] = exploration_bonus
        
        # Update previous score for next step (keep as float64 for precision)
        self._previous_score = np.float64(current_score)
        
        self.current_step += 1
        
        # Check termination conditions
        # Episode ends ONLY when:
        # 1. Max steps reached (500 steps) - FIXED episode length for long-horizon exploration
        # 2. No valid actions available (all actions would break contiguity)
        # NOTE: Illegal/disconnected states do NOT terminate - agent must learn recovery
        valid_mask = self.get_valid_action_mask()
        no_valid_actions = valid_mask.sum() == 0
        done = self.current_step >= self.max_steps or no_valid_actions
        
        # Determine if map is disconnected or illegal (from info dict set earlier)
        is_disconnected = info.get('is_disconnected', False)
        is_illegal = info.get('is_illegal', False) or max_pop_deviation > 5.0
        
        # Update info dict with all required fields
        info.update({
            "action": action,
            "node": node,
            "neighbor": neighbor,
            "old_district": old_district,
            "new_district": new_district,
            "step": self.current_step,
            "valid_actions": int(valid_mask.sum()),
            "reward": reward,
            "max_pop_deviation": max_pop_deviation,  # For early stopping
            "pop_deviations": pop_deviations,
            "efficiency_gap": efficiency_gap,  # For logging
            "is_disconnected": is_disconnected,
            "is_illegal": is_illegal,
            "total_score": float(current_score)  # Store total score in info for transparency
        })
        
        return self._get_observation(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        # SAFE INITIALIZATION: Always reset to baseline Arizona map (CONG_DIST)
        # This ensures agent starts in a legal state (Dev < 5%) and learns to improve it
        # rather than trying to fix a broken random map
        self.graph, self.partition = build_precinct_graph(self.state, self.basepath)
        # build_precinct_graph already uses CONG_DIST from the shapefile, which is the baseline map
        
        # Reset baseline assignment (in case graph structure changed)
        self._baseline_assignment = dict(self.partition.assignment)
        
        self.current_step = 0
        
        # Reset previous score for delta reward calculation
        self._previous_score = None
        
        # Re-initialize geometry cache after reset
        self._precinct_geometry_cache = {}
        for node in self.graph.nodes():
            if 'geometry' in self.graph.nodes[node]:
                geom = self.graph.nodes[node]['geometry']
                self._precinct_geometry_cache[node] = {
                    'area': geom.area,
                    'perimeter': geom.length
                }
        
        # Reset caches
        self._node_feature_totals = None
        self._cached_graph_observation = None
        self._partition_hash = None
        
        self._valid_actions = self._generate_valid_actions()
        # Reset action space to initial size
        self.action_space = spaces.Discrete(self._initial_action_space_size)
        return self._get_observation()
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Districts: {len(self.partition.parts)}")
            print(f"Valid actions: {len(self._valid_actions)}")
            
            try:
                metrics_df = self.metrics_calc.calculate_metrics(
                    self.partition, include_geometry=True
                )
                metrics = metrics_df.iloc[0]
                print(f"Efficiency Gap: {metrics['EfficiencyGap']:.4f}")
                print(f"Partisan Prop: {metrics['PartisanProp']:.4f}")
                print(f"Polsby-Popper Avg: {metrics['PolPopperAvg']:.4f}")
            except:
                print("Metrics calculation failed")
    
    def close(self):
        """Clean up environment resources."""
        pass
