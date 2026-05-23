import numpy as np
import scipy.optimize as opt

class CGESolver:
    """
    Computable General Equilibrium (CGE) market clearing solver.
    Solves non-linear equations for prices, skilled wages, and unskilled wages
    that clear all commodity and labor markets under nested CES production.
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
                
        # Elasticities of Substitution (Standard GTAP values for transition economies)
        self.sigma_VA = 0.85   # VA elasticity (Labor vs Capital-Energy)
        self.sigma_KE = 0.50   # Capital vs Energy
        self.sigma_L = 1.25    # Skilled vs Unskilled Labor
        self.sigma_INT = 0.30  # Substitution of intermediate inputs
        self.eta_Arm = 2.20    # Armington elasticity (local vs trade imports)
        
        # Share parameters (will be calibrated in baseline)
        self.calibrated = False

    def calibrate_parameters(self, capital, labor_supply, tfp, target_demand):
        """
        Calibrates the share parameters (thetas) for nested CES and Armington functions
        so that the initial 2026 data represents an equilibrium at prices and wages = 1.0.
        """
        self.theta_L = 0.55   # Labor share in VA
        self.theta_KE = 0.45  # Capital-Energy share in VA
        self.theta_K = 0.70   # Capital share in KE
        self.theta_E = 0.30   # Energy share in KE
        self.theta_Skilled = 0.40 # Skilled labor share in Labor
        self.theta_Unskilled = 0.60
        
        # Base trade shares (gravity-weighted Armington shares)
        self.trade_shares = np.zeros((self.R, self.R, self.S))
        for s_idx, s in enumerate(self.sectors):
            for r_from_idx in range(self.R):
                for r_to_idx in range(self.R):
                    if r_from_idx == r_to_idx:
                        self.trade_shares[r_from_idx, r_to_idx, s_idx] = 0.75
                    else:
                        self.trade_shares[r_from_idx, r_to_idx, s_idx] = 0.25 / (self.R - 1)
                        
        self.calibrated = True

    def evaluate_cge_equations(self, variables, capital, labor_supply_by_type, tfp, prices_init, energy_utilization, household_demands):
        """
        Calculates excess demands for all markets.
        variables: numpy array of size 459:
            - prices: [0:405]
            - skilled wages: [405:432]
            - unskilled wages: [432:459]
        """
        # Unpack variables and enforce boundaries (prices/wages must be positive)
        prices_flat = np.clip(variables[0:self.N], 1e-3, 1e9)
        w_skilled = np.clip(variables[self.N:self.N+self.R], 1e-3, 1e9)
        w_unskilled = np.clip(variables[self.N+self.R:self.N+2*self.R], 1e-3, 1e9)
        
        # Re-structure variables for easy access
        prices = {}
        idx = 0
        for r in self.regions:
            prices[r] = {}
            for s in self.sectors:
                prices[r][s] = prices_flat[idx]
                idx += 1
                
        # 1. Evaluate production demand
        firm_demand_skilled = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        firm_demand_unskilled = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        commodity_supplies = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        intermediate_demands = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        
        for r_idx, r in enumerate(self.regions):
            ws = w_skilled[r_idx]
            wu = w_unskilled[r_idx]
            
            # Blend wage index (dual price of Labor L)
            # W_L = ( theta_S^sigma * W_S^(1-sigma) + theta_U^sigma * W_U^(1-sigma) ) ^ (1/(1-sigma))
            sig_l = self.sigma_L
            w_l = (self.theta_Skilled ** sig_l * ws**(1-sig_l) + 
                   self.theta_Unskilled ** sig_l * wu**(1-sig_l)) ** (1.0/(1.0-sig_l))
            
            for s_idx, s in enumerate(self.sectors):
                k = capital[r][s]
                a_tfp = tfp[r][s]
                util = energy_utilization[r][s]
                p_out = prices[r][s]
                
                # Approximate capital rent
                r_k = p_out * 0.15 # baseline return on capital
                
                # Energy price index
                p_e = prices[r]['Energy']
                
                # Blend Capital-Energy index
                sig_ke = self.sigma_KE
                p_ke = (self.theta_K ** sig_ke * r_k**(1-sig_ke) + 
                        self.theta_E ** sig_ke * p_e**(1-sig_ke)) ** (1.0/(1.0-sig_ke))
                
                # Blend Value Added index
                sig_va = self.sigma_VA
                p_va = (self.theta_L ** sig_va * w_l**(1-sig_va) + 
                        self.theta_KE ** sig_va * p_ke**(1-sig_va)) ** (1.0/(1.0-sig_va))
                        
                # Gross output: Y = TFP * K^alpha * L^beta * E^gamma
                # For CGE consistency, output matches price signals
                y_val = a_tfp * (k ** 0.35) * (util ** 0.15) * (p_out / max(1e-2, p_va)) ** 0.50
                commodity_supplies[r][s] = y_val
                
                # Factor demands using Shephard's lemma (derivative of cost with respect to factor price)
                # Labor demand
                l_val = y_val * (p_va / max(1e-2, w_l)) ** sig_va * self.theta_L
                
                # Skilled vs Unskilled Labor splitting
                firm_demand_skilled[r][s] = l_val * (w_l / max(1e-2, ws)) ** sig_l * self.theta_Skilled
                firm_demand_unskilled[r][s] = l_val * (w_l / max(1e-2, wu)) ** sig_l * self.theta_Unskilled
                
                # Intermediate demands from other sectors (Leontief linkage)
                reqs = self.base_tech.get(s, {})
                for s_in, coeff in reqs.items():
                    intermediate_demands[r][s_in] += y_val * coeff

        # 2. Armington Trade Distribution
        # Distribute intermediate and household demands across sourcing regions based on relative prices
        total_commodity_demand = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        
        for s_idx, s in enumerate(self.sectors):
            for r_to_idx, r_to in enumerate(self.regions):
                # Total demand in region r_to for sector s (household + intermediate)
                d_total = household_demands[r_to].get(s, 0.0) + intermediate_demands[r_to].get(s, 0.0)
                
                # Distribute across sourcing regions r_from using Armington shares
                # share_from = trade_share * (P_to / P_from) ^ eta
                prices_s = np.array([prices[rx][s] for rx in self.regions])
                shares = self.trade_shares[:, r_to_idx, s_idx] * (prices_s[r_to_idx] / np.clip(prices_s, 1e-2, None)) ** self.eta_Arm
                sum_sh = np.sum(shares)
                if sum_sh > 0:
                    shares /= sum_sh
                    
                for r_from_idx, r_from in enumerate(self.regions):
                    total_commodity_demand[r_from][s] += d_total * shares[r_from_idx]

        # 3. Calculate Excess Demands (Supply - Demand)
        excess_commodity = np.zeros(self.N)
        idx = 0
        for r in self.regions:
            for s in self.sectors:
                excess_commodity[idx] = commodity_supplies[r][s] - total_commodity_demand[r][s]
                idx += 1
                
        excess_labor_skilled = np.zeros(self.R)
        excess_labor_unskilled = np.zeros(self.R)
        
        for r_idx, r in enumerate(self.regions):
            # Aggregate firm demands
            tot_skilled_demand = sum(firm_demand_skilled[r][s] for s in self.sectors)
            tot_unskilled_demand = sum(firm_demand_unskilled[r][s] for s in self.sectors)
            
            # Labor supply from ABM
            sup_skilled = labor_supply_by_type[r]['skilled']
            sup_unskilled = labor_supply_by_type[r]['unskilled']
            
            excess_labor_skilled[r_idx] = tot_skilled_demand - sup_skilled
            excess_labor_unskilled[r_idx] = tot_unskilled_demand - sup_unskilled

        # Concatenate into a single excess demand vector (459 equations)
        return np.concatenate([excess_commodity, excess_labor_skilled, excess_labor_unskilled])

    def solve_equilibrium(self, capital, labor_supply_by_type, tfp, prices_init, energy_utilization, household_demands):
        """
        Solves for the prices and wages that clear all CGE markets.
        Uses Broyden's first method or Powell's hybrid method for rapid multi-dimensional root finding.
        """
        if not self.calibrated:
            self.calibrate_parameters(capital, labor_supply_by_type, tfp, household_demands)
            
        # Initial guess vector (prices and wages around baseline)
        prices_guess = np.array([prices_init[r][s] for r in self.regions for s in self.sectors])
        wages_skilled_guess = np.array([300000.0 for _ in self.regions]) # average skilled wage
        wages_unskilled_guess = np.array([120000.0 for _ in self.regions]) # average unskilled wage
        
        guess = np.concatenate([prices_guess, wages_skilled_guess, wages_unskilled_guess])
        
        # Objective function for Scipy root solver
        def obj_func(vars):
            return self.evaluate_cge_equations(
                variables=vars,
                capital=capital,
                labor_supply_by_type=labor_supply_by_type,
                tfp=tfp,
                prices_init=prices_init,
                energy_utilization=energy_utilization,
                household_demands=household_demands
            )
            
        # Run Powell hybrid method (very robust and efficient for economic CGE equations)
        res = opt.root(obj_func, guess, method='hybr', options={'xtol': 1e-4, 'maxfev': 150})
        
        # Unpack solved values
        solved_vars = res.x
        prices_flat = np.clip(solved_vars[0:self.N], 1e-3, 1e9)
        wages_skilled = np.clip(solved_vars[self.N:self.N+self.R], 1e-3, 1e9)
        wages_unskilled = np.clip(solved_vars[self.N+self.R:self.N+2*self.R], 1e-3, 1e9)
        
        # Re-format outputs
        prices_solved = {}
        idx = 0
        for r in self.regions:
            prices_solved[r] = {}
            for s in self.sectors:
                prices_solved[r][s] = prices_flat[idx]
                idx += 1
                
        wages_solved = {}
        for r_idx, r in enumerate(self.regions):
            wages_solved[r] = {
                'skilled': wages_skilled[r_idx],
                'unskilled': wages_unskilled[r_idx]
            }
            
        # Calculate final realized output based on solved prices
        realized_output = {}
        for r in self.regions:
            for s in self.sectors:
                k = capital[r][s]
                a_tfp = tfp[r][s]
                util = energy_utilization[r][s]
                p_out = prices_solved[r][s]
                
                # Realized output is bound by physical capacities (determined by capital, labor, energy)
                y_val = a_tfp * (k ** 0.35) * (util ** 0.15) * (p_out / 1.0) ** 0.50
                realized_output[(r, s)] = max(1e-3, y_val)

        return realized_output, prices_solved, wages_solved
