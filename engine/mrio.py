import numpy as np
import scipy.linalg as la

class MRIOSolver:
    """
    Multi-Regional Input-Output (MRIO) Leontief solver.
    Tracks inter-sectoral and inter-regional trade flows for 27 regions and N sectors.
    Solves the equation: X = (I - A)^-1 * Y, under capacity constraints.
    Supports real coordinate distances for gravity-based trade shares calibration.
    """
    def __init__(self, regions, sectors, base_tech_coefficients, distances=None):
        """
        regions: list of 27 regions
        sectors: list of sectors
        base_tech_coefficients: dict of sector -> {input_sector: coefficient}
        distances: coordinates-based distance matrix (optional)
        """
        self.regions = regions
        self.sectors = sectors
        self.R = len(regions)
        self.S = len(sectors)
        self.N = self.R * self.S
        
        self.base_tech = base_tech_coefficients
        self.distances = distances
        
        # Build index mapping
        self.node_to_idx = {}
        self.idx_to_node = []
        idx = 0
        for r in self.regions:
            for s in self.sectors:
                self.node_to_idx[(r, s)] = idx
                self.idx_to_node.append((r, s))
                idx += 1
                
        # Initialize the global A matrix (N x N)
        self.A = self._build_initial_A()

    def _build_initial_A(self):
        """
        Constructs the large direct requirements matrix A (N x N).
        Combines technical sector needs with spatial trade weights.
        Trade flows are distributed using a gravity model:
        Region r's sector s imports from region q's sector s based on coordinate distances.
        """
        A = np.zeros((self.N, self.N))
        
        # Distance decay matrix for gravity-based trade distribution
        trade_gravity = np.zeros((self.R, self.R))
        for i in range(self.R):
            for j in range(self.R):
                if i == j:
                    trade_gravity[i, j] = 1.0 # High self-consumption (home bias)
                else:
                    dist = self.distances[i, j] if self.distances is not None else (100.0 + abs(i - j) * 35.0)
                    trade_gravity[i, j] = 0.2 / (1.0 + dist * 0.001) # Distance decay
                    
        # Normalize trade rows (how each region sources its inputs for a given sector)
        for s_idx in range(self.S):
            for r_from_idx in range(self.R):
                row_sum = np.sum(trade_gravity[:, r_from_idx])
                if row_sum > 0:
                    trade_gravity[:, r_from_idx] /= row_sum
                    
        # Populate A matrix
        # A[idx_input, idx_output]
        for r_out in self.regions:
            r_out_idx = self.regions.index(r_out)
            for s_out in self.sectors:
                out_node = (r_out, s_out)
                idx_out = self.node_to_idx[out_node]
                
                # Check what inputs s_out requires
                reqs = self.base_tech.get(s_out, {})
                for s_in, tech_coeff in reqs.items():
                    if s_in not in self.sectors:
                        continue
                    # Distribute s_in requirements across all sourcing regions
                    for r_in in self.regions:
                        r_in_idx = self.regions.index(r_in)
                        in_node = (r_in, s_in)
                        idx_in = self.node_to_idx[in_node]
                        
                        # Trade share determines how much of region r_out's sector s_out input
                        # comes from region r_in's sector s_in.
                        trade_share = trade_gravity[r_in_idx, r_out_idx]
                        A[idx_in, idx_out] = tech_coeff * trade_share
                        
        return A

    def balance_A_matrix_ras(self, target_row_sums, target_col_sums, max_iter=100, tolerance=1e-5):
        """
        Applies the RAS biproportional scaling method to adjust technical coefficients
        when regional production structure shifts, ensuring matrix matches observed totals.
        """
        A_balanced = self.A.copy()
        
        for _ in range(max_iter):
            # Row scaling
            row_sums = np.sum(A_balanced, axis=1)
            row_sums[row_sums == 0] = 1.0
            r_scale = target_row_sums / row_sums
            A_balanced = (A_balanced.T * r_scale).T
            
            # Column scaling
            col_sums = np.sum(A_balanced, axis=0)
            col_sums[col_sums == 0] = 1.0
            s_scale = target_col_sums / col_sums
            A_balanced = A_balanced * s_scale
            
            # Check convergence
            curr_row_sums = np.sum(A_balanced, axis=1)
            if np.allclose(curr_row_sums, target_row_sums, atol=tolerance):
                break
                
        self.A = A_balanced

    def solve_equilibrium(self, final_demand):
        """
        Solves X = (I - A)^-1 * Y
        final_demand: dict of (region, sector) -> final demand value
        Returns:
            X: numpy array of required total outputs (size N)
        """
        Y = np.zeros(self.N)
        for node, val in final_demand.items():
            if node in self.node_to_idx:
                Y[self.node_to_idx[node]] = val
                
        I = np.eye(self.N)
        try:
            X = la.solve(I - self.A, Y)
        except la.LinAlgError:
            # Fallback if singular matrix occurs (very rare with normalized coefficients)
            # Use pseudoinverse solver
            X = np.linalg.pinv(I - self.A) @ Y
            
        # Ensure outputs are non-negative
        X = np.clip(X, 0.0, None)
        return X

    def resolve_bottlenecks(self, target_final_demand, production_capacities, max_iter=5):
        """
        Resolves bottlenecks when required output (Leontief) exceeds actual physical production capacity
        (which is limited by Capital, Labor, or Energy).
        
        Applies an iterative rationing algorithm to cascade supply constraints through the trade matrix.
        """
        Y_realized = target_final_demand.copy()
        
        # Handle both flat (region, sector) -> val and nested region -> sector -> val capacities
        X_cap = np.zeros(self.N)
        for k, v in production_capacities.items():
            if isinstance(v, dict):
                r = k
                for s, val in v.items():
                    node = (r, s)
                    if node in self.node_to_idx:
                        X_cap[self.node_to_idx[node]] = val
            else:
                node = k
                if node in self.node_to_idx:
                    X_cap[self.node_to_idx[node]] = v
                
        # First pass to calculate initial request ratios
        X_first = self.solve_equilibrium(target_final_demand)
        initial_ratios = np.zeros(self.N)
        for i in range(self.N):
            if X_first[i] > 0:
                initial_ratios[i] = X_first[i] / max(1e-5, X_cap[i])

        # Iterative rationing
        for _ in range(max_iter):
            X_req = self.solve_equilibrium(Y_realized)
            
            # Find any active bottlenecks
            ratios = np.ones(self.N)
            has_bottleneck = False
            for i in range(self.N):
                if X_req[i] > X_cap[i] and X_req[i] > 0:
                    ratios[i] = X_req[i] / max(1e-5, X_cap[i])
                    has_bottleneck = True
            
            if not has_bottleneck:
                break
                
            # Scale down final demand for bottlenecked nodes
            for node, idx in self.node_to_idx.items():
                if ratios[idx] > 1.0:
                    Y_realized[node] = Y_realized[node] / ratios[idx]
                    
        # Final solved output under scaled demand
        X_realized = self.solve_equilibrium(Y_realized)
        X_realized = np.minimum(X_realized, X_cap) # Hard ceiling at capacity
        
        # Format results
        realized_output_dict = {}
        bottlenecks = {}
        for node, idx in self.node_to_idx.items():
            realized_output_dict[node] = X_realized[idx]
            # Report the initial demand shortage ratio (so users know where the bottlenecks were)
            bottlenecks[node] = initial_ratios[idx]
            
        return realized_output_dict, Y_realized, bottlenecks
