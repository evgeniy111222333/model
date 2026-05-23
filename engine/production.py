import numpy as np

class ProductionEngine:
    """
    Handles sectoral production functions, capital accumulation, and labor allocation
    across 27 regions and 15 sectors.
    """
    def __init__(self, regions, sectors, alpha=0.35, beta=0.50, gamma=0.15, depreciation_rates=None):
        """
        alpha: capital elasticity of output
        beta: labor elasticity of output
        gamma: energy elasticity of output
        Note: alpha + beta + gamma = 1.0 (constant returns to scale)
        """
        self.regions = regions
        self.sectors = sectors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Base depreciation rates (e.g. 5% to 8% annually)
        if depreciation_rates is None:
            self.depreciation = {r: {s: 0.06 for s in sectors} for r in regions}
        else:
            self.depreciation = depreciation_rates

    def accumulate_capital(self, capital, investment, war_damage):
        """
        K(t+1) = (1 - delta - war_damage) * K(t) + I(t)
        """
        next_capital = {}
        for r in self.regions:
            next_capital[r] = {}
            for s in self.sectors:
                k_curr = capital[r][s]
                inv = investment[r][s]
                dmg = war_damage.get(r, {}).get(s, 0.0)
                dep = self.depreciation[r][s]
                
                # Capital cannot fall below a tiny positive value to prevent division by zero in Cobb-Douglas
                next_capital[r][s] = max(1e-3, k_curr * (1.0 - dep - dmg) + inv)
        return next_capital

    def solve_labor_allocation(self, capital, labor_supply, tfp, prices, energy_utilization):
        """
        Solves for regional wage rates and allocates labor across sectors such that
        total regional labor demand matches regional labor supply.
        
        Using Cobb-Douglas marginal product of labor:
        MPL_r,s = beta * Y_r,s / L_r,s
        In equilibrium: Wage_r = Price_r,s * MPL_r,s
        Since L_r,s = Price_r,s * beta * A_r,s * K_r,s^alpha * E_r,s^gamma * L_r,s^(beta-1) * Util_r,s
        We can solve for L_r,s in terms of Wage_r:
        L_r,s = ( (Price_r,s * beta * A_r,s * K_r,s^alpha * E_r,s^gamma * Util_r,s) / Wage_r ) ^ (1 / (1 - beta))
        
        We iterate using Newton's method (or simple bisection per region) to find Wage_r
        such that sum_s(L_r,s(Wage_r)) = labor_supply_r.
        """
        allocated_labor = {}
        regional_wages = {}
        sectoral_output = {}
        
        for r in self.regions:
            l_sup = labor_supply[r]
            if l_sup <= 0:
                # Ghost region or completely evacuated
                allocated_labor[r] = {s: 0.0 for s in self.sectors}
                regional_wages[r] = 0.0
                sectoral_output[r] = {s: 0.0 for s in self.sectors}
                continue
                
            # Sector-specific wage attraction variables
            # A_term = Price * beta * TFP * K^alpha * E^gamma * Util
            a_terms = {}
            for s in self.sectors:
                p = prices[r][s]
                a_tfp = tfp[r][s]
                k = capital[r][s]
                e_util = energy_utilization[r][s]
                
                # Approximate baseline energy input as proportional to capital and TFP
                e_input = (k ** 0.5) * e_util
                
                a_terms[s] = p * self.beta * a_tfp * (k ** self.alpha) * (e_input ** self.gamma)

            # Solve for Wage_r: sum_s (a_terms[s] / wage) ^ (1 / (1 - beta)) = l_sup
            # Let exponent = 1 / (1 - beta)
            exponent = 1.0 / (1.0 - self.beta)
            
            # Simple Brent-like bisection solver for Wage_r
            low_w, high_w = 1e-5, 1e7
            w_est = 1.0
            
            for _ in range(50):
                w_est = (low_w + high_w) / 2.0
                l_demands = {s: (a_terms[s] / w_est) ** exponent for s in self.sectors}
                tot_demand = sum(l_demands.values())
                
                if abs(tot_demand - l_sup) < 1e-4 * l_sup:
                    break
                elif tot_demand > l_sup:
                    low_w = w_est
                else:
                    high_w = w_est
            
            regional_wages[r] = w_est
            allocated_labor[r] = {}
            sectoral_output[r] = {}
            
            # Allocate labor and calculate output
            for s in self.sectors:
                # Realized labor demand
                l_val = max(1e-4, (a_terms[s] / w_est) ** exponent)
                allocated_labor[r][s] = l_val
                
                # Calculate output: Y = TFP * K^alpha * L^beta * E^gamma * Util
                a_tfp = tfp[r][s]
                k = capital[r][s]
                e_util = energy_utilization[r][s]
                e_input = (k ** 0.5) * e_util
                
                y_val = a_tfp * (k ** self.alpha) * (l_val ** self.beta) * (e_input ** self.gamma)
                sectoral_output[r][s] = y_val
                
        return allocated_labor, regional_wages, sectoral_output
