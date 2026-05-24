import numpy as np
import scipy.optimize as opt

class CGESolver:
    """
    Computable General Equilibrium (CGE) market clearing solver.
    Solves 486 non-linear equations (405 commodity prices + 27 * 3 labor wages)
    using a fully vectorized NumPy matrix representation to bypass Python loops.
    """
    def __init__(self, regions, sectors, base_tech_coefficients):
        self.regions = regions
        self.sectors = sectors
        self.R = len(regions)
        self.S = len(sectors)
        self.N = self.R * self.S
        
        self.base_tech = base_tech_coefficients
        
        # Mapping index
        self.node_to_idx = {}
        self.idx_to_node = []
        idx = 0
        for r in self.regions:
            for s in self.sectors:
                self.node_to_idx[(r, s)] = idx
                self.idx_to_node.append((r, s))
                idx += 1
                
        # Build direct requirement coefficient matrix B (15 x 15)
        # B[i, j] = how much of sector i is needed per unit of output of sector j
        self.B_mat = np.zeros((self.S, self.S))
        for s_idx, s in enumerate(self.sectors):
            reqs = self.base_tech.get(s, {})
            for s_in, coeff in reqs.items():
                s_in_idx = self.sectors.index(s_in)
                self.B_mat[s_in_idx, s_idx] = coeff
                
        # Elasticities of Substitution
        self.sigma_VA = 0.85   # Value Added: Labor vs Capital-Energy
        self.sigma_KE = 0.50   # Capital vs Energy
        self.sigma_L = 1.25    # Labor types: Skilled vs Semi-skilled vs Unskilled
        self.sigma_INT = 0.30  # Intermediate inputs
        self.eta_Arm = 2.20    # Armington Regional Trade
        self.eta_World = 1.80  # Armington World Import
        
        # CES Share parameters
        self.theta_u = 0.50
        self.theta_m = 0.35
        self.theta_s = 0.15
        
        self.theta_L = 0.55
        self.theta_KE = 0.45
        
        self.theta_K = 0.70
        self.theta_E = 0.30
        
        self.calibrated = False
        self.trade_shares = None
        self.distances = None

    def _init_distances(self):
        self.distances = np.zeros((self.R, self.R))
        for i in range(self.R):
            for j in range(self.R):
                if i == j:
                    self.distances[i, j] = 1.0
                else:
                    self.distances[i, j] = 100.0 + abs(i - j) * 35.0

    def calibrate_parameters(self, capital, labor_supply, tfp, target_demand):
        """
        Calibrates trade shares using gravity formulation.
        """
        if self.distances is None:
            self._init_distances()
            
        self.trade_shares = np.zeros((self.R, self.R, self.S))
        
        for s_idx, s in enumerate(self.sectors):
            for r_dest_idx in range(self.R):
                weights = np.zeros(self.R)
                for r_src_idx in range(self.R):
                    dist = self.distances[r_src_idx, r_dest_idx]
                    cap_src = sum(capital[self.regions[r_src_idx]].values())
                    weights[r_src_idx] = cap_src / (dist ** 1.3)
                    
                weights[r_dest_idx] *= 5.0 # high home bias
                
                sum_w = np.sum(weights)
                if sum_w > 0:
                    weights /= sum_w
                else:
                    weights = np.zeros(self.R)
                    weights[r_dest_idx] = 1.0
                    
                self.trade_shares[:, r_dest_idx, s_idx] = weights
                
        self.calibrated = True

    def evaluate_cge_equations(self, multipliers, capital_mat, labor_supply_mat, tfp_mat, prices_base_mat, wages_base_mat, energy_util_mat, hh_demands_mat):
        """
        Evaluates excess demands using fully vectorized matrix algebra.
        """
        # Multipliers clipped for numerical bounds
        mult_clip = np.clip(multipliers, 1e-3, 1e9)
        prices_mult = mult_clip[0:self.N].reshape((self.R, self.S))
        wages_skilled_mult = mult_clip[self.N:self.N+self.R]
        wages_semiskilled_mult = mult_clip[self.N+self.R:self.N+2*self.R]
        wages_unskilled_mult = mult_clip[self.N+2*self.R:self.N+3*self.R]
        
        # Prices (R, S) and Wages (R, 3)
        prices = prices_base_mat * prices_mult
        
        w_unskilled = wages_base_mat[:, 0] * wages_unskilled_mult
        w_semiskilled = wages_base_mat[:, 1] * wages_semiskilled_mult
        w_skilled = wages_base_mat[:, 2] * wages_skilled_mult
        
        # 1. Labor cost index: w_L (R,)
        sig_L = self.sigma_L
        w_L = (self.theta_u ** sig_L * w_unskilled**(1-sig_L) + 
               self.theta_m ** sig_L * w_semiskilled**(1-sig_L) + 
               self.theta_s ** sig_L * w_skilled**(1-sig_L)) ** (1.0/(1.0-sig_L))
        
        # Capital rent (R, S)
        rk = prices * 0.15
        
        # Energy price in each region (R,)
        pe = prices[:, 3] # Energy is index 3 in SECTORS
        
        # 2. Capital-Energy index: p_KE (R, S)
        sig_KE = self.sigma_KE
        pe_bc = pe[:, np.newaxis]
        p_KE = (self.theta_K ** sig_KE * rk**(1-sig_KE) + 
                self.theta_E ** sig_KE * pe_bc**(1-sig_KE)) ** (1.0/(1.0-sig_KE))
        
        # 3. Value Added index: p_VA (R, S)
        sig_VA = self.sigma_VA
        w_L_bc = w_L[:, np.newaxis]
        p_VA = (self.theta_L ** sig_VA * w_L_bc**(1-sig_VA) + 
                self.theta_KE ** sig_VA * p_KE**(1-sig_VA)) ** (1.0/(1.0-sig_VA))
        
        # 4. Production Output: Y (R, S)
        cap_factor = (capital_mat ** 0.50) * energy_util_mat
        y_val = tfp_mat * cap_factor * (prices / np.clip(p_VA, 1e-2, None)) ** 0.50
        
        # 5. Factor Demands
        L_composite = y_val * (p_VA / np.clip(w_L_bc, 1e-2, None)) ** sig_VA * self.theta_L
        
        wu_bc = w_unskilled[:, np.newaxis]
        wm_bc = w_semiskilled[:, np.newaxis]
        ws_bc = w_skilled[:, np.newaxis]
        
        L_u = L_composite * (w_L_bc / np.clip(wu_bc, 1e-2, None)) ** sig_L * self.theta_u
        L_m = L_composite * (w_L_bc / np.clip(wm_bc, 1e-2, None)) ** sig_L * self.theta_m
        L_s = L_composite * (w_L_bc / np.clip(ws_bc, 1e-2, None)) ** sig_L * self.theta_s
        
        dem_unskilled = np.sum(L_u, axis=1)
        dem_semiskilled = np.sum(L_m, axis=1)
        dem_skilled = np.sum(L_s, axis=1)
        
        # 6. Intermediate Demands (R, S)
        # B_mat is (S, S), y_val is (R, S), so y_val @ B_mat.T represents required intermediate inputs
        intermediate_demands = y_val @ self.B_mat.T
        
        # 7. Armington Trade Distribution
        d_total = hh_demands_mat + intermediate_demands
        d_domestic = d_total * 0.85
        
        total_commodity_demand = np.zeros((self.R, self.S))
        
        for s_idx in range(self.S):
            # shares is (R, R) -> from src (row) to dest (col)
            shares = self.trade_shares[:, :, s_idx]
            p_src = prices[:, s_idx]
            
            # composite price in each destination region (R,)
            p_comp = shares.T @ p_src
            
            # Adjust trade shares based on relative prices
            shares_adj = shares * (p_comp / np.clip(p_src[:, np.newaxis], 1e-2, None)) ** self.eta_Arm
            
            # Normalize trade shares per destination column
            sum_shares = np.sum(shares_adj, axis=0)
            sum_shares[sum_shares == 0] = 1.0
            shares_adj /= sum_shares
            
            total_commodity_demand[:, s_idx] = shares_adj @ d_domestic[:, s_idx]

        # 8. Excess Demands
        excess_commodity = (total_commodity_demand - y_val).flatten()
        excess_labor_skilled = dem_skilled - labor_supply_mat[:, 2]
        excess_labor_semiskilled = dem_semiskilled - labor_supply_mat[:, 1]
        excess_labor_unskilled = dem_unskilled - labor_supply_mat[:, 0]
        
        return np.concatenate([
            excess_commodity, 
            excess_labor_skilled, 
            excess_labor_semiskilled, 
            excess_labor_unskilled
        ])

    def solve_equilibrium(self, capital, labor_supply_by_type, tfp, prices_init, energy_utilization, household_demands):
        """
        Solves for the prices and wages that clear all CGE markets.
        Uses Broyden's first method or Powell's hybrid method for rapid multi-dimensional root finding.
        """
        if not self.calibrated:
            self.calibrate_parameters(capital, labor_supply_by_type, tfp, household_demands)
            
        # Convert all input dictionary variables to NumPy arrays (R, S) or (R, 3)
        capital_mat = np.zeros((self.R, self.S))
        tfp_mat = np.zeros((self.R, self.S))
        prices_init_mat = np.zeros((self.R, self.S))
        energy_util_mat = np.zeros((self.R, self.S))
        hh_demands_mat = np.zeros((self.R, self.S))
        
        for r_idx, r in enumerate(self.regions):
            for s_idx, s in enumerate(self.sectors):
                capital_mat[r_idx, s_idx] = capital[r][s]
                tfp_mat[r_idx, s_idx] = tfp[r][s]
                prices_init_mat[r_idx, s_idx] = prices_init[r][s]
                energy_util_mat[r_idx, s_idx] = energy_utilization[r][s]
                hh_demands_mat[r_idx, s_idx] = household_demands[r][s]
                
        labor_supply_mat = np.zeros((self.R, 3))
        wages_base_mat = np.zeros((self.R, 3))
        
        for r_idx, r in enumerate(self.regions):
            labor_supply_mat[r_idx, 0] = labor_supply_by_type[r]['unskilled']
            labor_supply_mat[r_idx, 1] = labor_supply_by_type[r].get('semi-skilled', labor_supply_by_type[r]['unskilled'] * 0.70)
            labor_supply_mat[r_idx, 2] = labor_supply_by_type[r]['skilled']
            
            # Baseline wages
            wages_base_mat[r_idx, 0] = labor_supply_by_type[r].get('unskilled_wage', 120000.0)
            wages_base_mat[r_idx, 2] = labor_supply_by_type[r].get('skilled_wage', 300000.0)
            wages_base_mat[r_idx, 1] = (wages_base_mat[r_idx, 0] + wages_base_mat[r_idx, 2]) / 2.0
            
        guess = np.ones(self.N + 3 * self.R)
        
        def obj_func(vars):
            return self.evaluate_cge_equations(
                multipliers=vars,
                capital_mat=capital_mat,
                labor_supply_mat=labor_supply_mat,
                tfp_mat=tfp_mat,
                prices_base_mat=prices_init_mat,
                wages_base_mat=wages_base_mat,
                energy_util_mat=energy_util_mat,
                hh_demands_mat=hh_demands_mat
            )
            
        # Run Powell hybrid method
        res = opt.root(obj_func, guess, method='hybr', options={'xtol': 1e-4, 'maxfev': 150})
        
        # Unpack solved multipliers and clip
        solved_mult = np.clip(res.x, 1e-3, 1e9)
        prices_mult = solved_mult[0:self.N].reshape((self.R, self.S))
        wages_skilled_mult = solved_mult[self.N:self.N+self.R]
        wages_semiskilled_mult = solved_mult[self.N+self.R:self.N+2*self.R]
        wages_unskilled_mult = solved_mult[self.N+2*self.R:self.N+3*self.R]
        
        # Re-construct wages and prices
        prices_solved = {}
        for r_idx, r in enumerate(self.regions):
            prices_solved[r] = {}
            for s_idx, s in enumerate(self.sectors):
                prices_solved[r][s] = prices_init_mat[r_idx, s_idx] * prices_mult[r_idx, s_idx]
                
        wages_solved = {}
        for r_idx, r in enumerate(self.regions):
            wages_solved[r] = {
                'unskilled': wages_base_mat[r_idx, 0] * wages_unskilled_mult[r_idx],
                'semi-skilled': wages_base_mat[r_idx, 1] * wages_semiskilled_mult[r_idx],
                'skilled': wages_base_mat[r_idx, 2] * wages_skilled_mult[r_idx]
            }
            
        # Re-evaluate final gross outputs
        prices_solved_mat = prices_init_mat * prices_mult
        rk = prices_solved_mat * 0.15
        pe = prices_solved_mat[:, 3]
        pe_bc = pe[:, np.newaxis]
        
        sig_KE = self.sigma_KE
        p_KE = (self.theta_K ** sig_KE * rk**(1-sig_KE) + 
                self.theta_E ** sig_KE * pe_bc**(1-sig_KE)) ** (1.0/(1.0-sig_KE))
        
        wu = wages_base_mat[:, 0] * wages_unskilled_mult
        wm = wages_base_mat[:, 1] * wages_semiskilled_mult
        ws = wages_base_mat[:, 2] * wages_skilled_mult
        sig_L = self.sigma_L
        w_L = (self.theta_u ** sig_L * wu**(1-sig_L) + 
               self.theta_m ** sig_L * wm**(1-sig_L) + 
               self.theta_s ** sig_L * ws**(1-sig_L)) ** (1.0/(1.0-sig_L))
        
        sig_VA = self.sigma_VA
        w_L_bc = w_L[:, np.newaxis]
        p_VA = (self.theta_L ** sig_VA * w_L_bc**(1-sig_VA) + 
                self.theta_KE ** sig_VA * p_KE**(1-sig_VA)) ** (1.0/(1.0-sig_VA))
        
        cap_factor = (capital_mat ** 0.50) * energy_util_mat
        y_val = tfp_mat * cap_factor * (prices_solved_mat / np.clip(p_VA, 1e-2, None)) ** 0.50
        
        realized_output = {}
        for r_idx, r in enumerate(self.regions):
            for s_idx, s in enumerate(self.sectors):
                realized_output[(r, s)] = max(1e-3, y_val[r_idx, s_idx])
                
        return realized_output, prices_solved, wages_solved
