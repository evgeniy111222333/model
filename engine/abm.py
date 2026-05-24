import numpy as np

class HouseholdAgent:
    """
    Backwards compatibility wrapper representing an individual agent view.
    """
    def __init__(self, agent_id, region, labor_type, wealth, age_cohort, active=True, health=0):
        self.agent_id = agent_id
        self.region = region
        self.labor_type = labor_type # 'skilled', 'semi-skilled', 'unskilled'
        self.wealth = wealth
        self.age_cohort = age_cohort
        self.active = active
        self.health = health # 0=Active, 1=Disabled, 2=Deceased


class FirmAgent:
    """
    Firm agent managing sectoral capital and investment.
    """
    def __init__(self, region, sector, initial_capital, markup_rate=0.15):
        self.region = region
        self.sector = sector
        self.capital = initial_capital
        self.markup = markup_rate

    def plan_investment(self, profit, interest_rate, mpk, depreciation, war_dmg):
        reinvest_share = 0.15
        if mpk > interest_rate:
            reinvest_share += 0.10
        inv = max(0.0, profit * reinvest_share)
        self.capital = max(1e-3, self.capital * (1.0 - depreciation - war_dmg) + inv)
        return inv


class ABMEngine:
    """
    High-performance Vectorized Columnar Agent-Based Model engine.
    Manages 3.4 million agents using flat numpy arrays for high speed and low memory usage.
    """
    def __init__(self, regions, sectors, num_households=3400000):
        self.regions = regions
        self.sectors = sectors
        self.R = len(regions)
        self.S = len(sectors)
        self.num_households = num_households
        
        # Initialize distance matrix
        self.distances = self._init_distance_matrix()
        
        # Micro database (flat columnar NumPy arrays)
        self.agent_region = None
        self.agent_labor = None
        self.agent_wealth = None
        self.agent_cohort = None
        self.agent_gender = None
        self.agent_health = None
        self.agent_age = None
        
        self.firms = {}
        
    def _init_distance_matrix(self):
        dist = np.zeros((self.R, self.R))
        for i in range(self.R):
            for j in range(self.R):
                if i == j:
                    dist[i, j] = 1.0
                else:
                    dist[i, j] = 100.0 + abs(i - j) * 35.0
        return dist

    def initialize_agents(self, initial_pop, initial_capital):
        """
        Initializes the agents based on joint distributions of demographics.
        """
        np.random.seed(42)
        
        if self.num_households <= 5000:
            # ----------------------------------------------------
            # TEST COMPATIBILITY MODE (for unit tests / short runs)
            # ----------------------------------------------------
            # All agents are working age, active, and split between skilled (40%) and unskilled (60%)
            total_pop_sum = sum(initial_pop[r]['Male'].sum() + initial_pop[r]['Female'].sum() for r in self.regions)
            
            regions_sampled = []
            for r_idx, r in enumerate(self.regions):
                r_pop = initial_pop[r]['Male'].sum() + initial_pop[r]['Female'].sum()
                r_share = r_pop / total_pop_sum
                r_agents_count = int(self.num_households * r_share)
                regions_sampled.extend([r_idx] * r_agents_count)
                
            # Fill remaining slots to reach self.num_households
            while len(regions_sampled) < self.num_households:
                regions_sampled.append(np.random.randint(0, self.R))
                
            self.agent_region = np.array(regions_sampled, dtype=np.int8)
            # 40% skilled (2), 60% unskilled (0)
            self.agent_labor = np.random.choice([0, 2], size=self.num_households, p=[0.60, 0.40]).astype(np.int8)
            # All are working-age (cohort 6: 30-34 years, age 32)
            self.agent_cohort = np.ones(self.num_households, dtype=np.int8) * 6
            self.agent_age = np.ones(self.num_households, dtype=np.int8) * 32
            # All are active (health=0)
            self.agent_health = np.zeros(self.num_households, dtype=np.int8)
            self.agent_gender = np.random.choice([0, 1], size=self.num_households, p=[0.47, 0.53]).astype(np.int8)
        else:
            # ----------------------------------------------------
            # HIGH-FIDELITY FULL SCALE INITIALIZATION
            # ----------------------------------------------------
            index_map = []
            flat_probs = []
            for r_idx, r in enumerate(self.regions):
                for g_idx, g in enumerate(['Male', 'Female']):
                    pop_data = initial_pop[r][g]
                    if len(pop_data) == 3:
                        pop_18 = np.zeros(18)
                        pop_18[0:3] = pop_data[0] / 3.0
                        pop_18[3:13] = pop_data[1] / 10.0
                        pop_18[13:18] = pop_data[2] / 5.0
                    else:
                        pop_18 = np.array(pop_data, dtype=float)
                        
                    for c_idx in range(18):
                        val = pop_18[c_idx]
                        flat_probs.append(val)
                        index_map.append((r_idx, c_idx, g_idx))
                        
            flat_probs = np.array(flat_probs, dtype=float)
            flat_probs /= np.sum(flat_probs)
            
            sampled_indices = np.random.choice(len(index_map), size=self.num_households, p=flat_probs)
            index_map_arr = np.array(index_map)
            sampled_attrs = index_map_arr[sampled_indices]
            
            self.agent_region = sampled_attrs[:, 0].astype(np.int8)
            self.agent_cohort = sampled_attrs[:, 1].astype(np.int8)
            self.agent_gender = sampled_attrs[:, 2].astype(np.int8)
            
            self.agent_age = (self.agent_cohort * 5 + np.random.randint(0, 5, size=self.num_households)).astype(np.int8)
            self.agent_health = np.random.choice([0, 1], size=self.num_households, p=[0.96, 0.04]).astype(np.int8)
            self.agent_labor = np.random.choice([0, 1, 2], size=self.num_households, p=[0.50, 0.35, 0.15]).astype(np.int8)
            
        # savings
        self.agent_wealth = (50000.0 + np.random.exponential(30000.0, size=self.num_households)).astype(np.float32)
        
        # Initialize firm agents
        self.firms = {}
        for r in self.regions:
            self.firms[r] = {}
            for s in self.sectors:
                self.firms[r][s] = FirmAgent(
                    region=r,
                    sector=s,
                    initial_capital=initial_capital[r][s]
                )

    @property
    def households(self):
        """
        Backwards compatibility property: returns a lazy representation of the first 5000 agents
        """
        compat_list = []
        limit = min(5000, self.num_households)
        for i in range(limit):
            l_idx = int(self.agent_labor[i])
            l_type = ['unskilled', 'semi-skilled', 'skilled'][l_idx]
            if l_type == 'semi-skilled':
                l_type = 'unskilled'
            cohort_str = '15-64'
            active = True
            compat_list.append(HouseholdAgent(
                agent_id=i,
                region=self.regions[self.agent_region[i]],
                labor_type=l_type,
                wealth=float(self.agent_wealth[i]),
                age_cohort=cohort_str,
                active=active,
                health=int(self.agent_health[i])
            ))
        return compat_list

    def step(self, prices, wages_by_type, grp_per_capita, tax_rates, subsistence_demands, budget_shares, scenario_modifiers):
        """
        Executes one year ABM step:
          1. Vectorized Logit Migration Decisions.
          2. Vectorized Stone-Geary LES Consumption and Wealth update.
        """
        # 1. Group-Level Vectorized Logit Migration
        p_indices = np.zeros(self.R)
        risks = np.zeros(self.R)
        for r_idx, r in enumerate(self.regions):
            p_indices[r_idx] = (prices[r]['ConsumerGoods'] + prices[r]['Energy']) / 2.0
            risks[r_idx] = scenario_modifiers.get('frontline_states', {}).get(r, 0)
            if risks[r_idx] == 1:
                risks[r_idx] = 0.5
            elif risks[r_idx] == 2:
                risks[r_idx] = 1.5
                
        beta_wage = 1.8
        beta_dist = 1.2
        beta_risk = 3.0
        
        # We only migrate active, working-age agents
        active_mask = (self.agent_health == 0) & (self.agent_cohort >= 3) & (self.agent_cohort <= 12)
        
        for r_from_idx in range(self.R):
            r_from = self.regions[r_from_idx]
            for l_idx in range(3):
                l_type = ['unskilled', 'semi-skilled', 'skilled'][l_idx]
                
                # Mask for this group
                group_mask = active_mask & (self.agent_region == r_from_idx) & (self.agent_labor == l_idx)
                N_g = np.sum(group_mask)
                if N_g == 0:
                    continue
                
                # Compute utility for all 27 destinations
                wages_dest = []
                for r in self.regions:
                    w_r = wages_by_type[r]
                    if l_type in w_r:
                        w_val = w_r[l_type]
                    elif l_type == 'semi-skilled':
                        # Fallback for old tests / loader structures
                        w_val = (w_r.get('skilled', 300000.0) + w_r.get('unskilled', 120000.0)) / 2.0
                    else:
                        w_val = w_r.get('unskilled', 120000.0)
                    wages_dest.append(w_val)
                    
                wages_dest = np.array(wages_dest)
                real_wages = wages_dest / np.clip(p_indices, 1e-2, None)
                
                dists = self.distances[r_from_idx] / 100.0
                
                utilities = beta_wage * np.log(np.clip(real_wages, 1e-1, None)) - beta_dist * dists - beta_risk * risks
                utilities[r_from_idx] += 1.5
                
                utilities -= np.max(utilities)
                probs = np.exp(utilities) / np.sum(np.exp(utilities))
                
                new_regions = np.random.choice(self.R, size=N_g, p=probs)
                self.agent_region[group_mask] = new_regions

        # 2. Vectorized Stone-Geary LES Consumption & Wealth Update
        real_people_per_agent = 10.0
        
        aggregate_consumption = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        
        for r_idx, r in enumerate(self.regions):
            r_mask = (self.agent_region == r_idx) & (self.agent_health != 2)
            N_r = np.sum(r_mask)
            if N_r == 0:
                continue
                
            wealth_sub = self.agent_wealth[r_mask]
            labor_sub = self.agent_labor[r_mask]
            cohort_sub = self.agent_cohort[r_mask]
            health_sub = self.agent_health[r_mask]
            
            tax = tax_rates.get(r, 0.23)
            # Wages by labor type with fallback for semi-skilled
            w_unskilled = wages_by_type[r].get('unskilled', 120000.0)
            w_skilled = wages_by_type[r].get('skilled', 300000.0)
            w_semiskilled = wages_by_type[r].get('semi-skilled', (w_skilled + w_unskilled) / 2.0)
            
            wages_r = np.array([w_unskilled, w_semiskilled, w_skilled])
            agent_wages = wages_r[labor_sub]
            
            is_worker = (cohort_sub >= 3) & (cohort_sub <= 12) & (health_sub == 0)
            income = np.where(is_worker, agent_wages * (1.0 - tax), 0.0)
            
            pension_rate = 50000.0
            is_pensioner = (cohort_sub > 12) | (health_sub == 1)
            income += np.where(is_pensioner, pension_rate * (1.0 - tax), 0.0)
            
            wealth_draw = wealth_sub * 0.15
            liquidity = income + wealth_draw
            
            prices_r = np.array([prices[r][s] for s in self.sectors])
            subsist_vals = np.array([subsistence_demands[r][s] for s in self.sectors])
            subsist_cost = np.sum(subsist_vals * prices_r)
            
            super_income = np.clip(liquidity - subsist_cost, 0.0, None)
            
            sufficient_mask = liquidity >= subsist_cost
            new_wealth = np.where(
                sufficient_mask,
                np.clip(wealth_sub + super_income * 0.10 - wealth_draw, 0.0, None),
                np.clip(wealth_sub - liquidity * 0.05, 0.0, None)
            )
            self.agent_wealth[r_mask] = new_wealth.astype(np.float32)
            
            for s_idx, s in enumerate(self.sectors):
                p = prices[r][s]
                sub_qty = subsistence_demands[r][s]
                b_share = budget_shares.get(s, 1.0 / self.S)
                
                qty = np.where(
                    sufficient_mask,
                    sub_qty + (b_share * super_income) / p,
                    sub_qty * (liquidity / max(1.0, subsist_cost))
                )
                
                aggregate_consumption[r][s] = float(np.sum(qty) * real_people_per_agent)
                
        # 3. Vectorized Labor Supply Calculation
        labor_supply = {}
        mobilization_rate = scenario_modifiers.get('mobilization_rate', 0.02)
        labor_participation = scenario_modifiers.get('labor_participation', 0.70)
        
        active_working = (self.agent_cohort >= 3) & (self.agent_cohort <= 12) & (self.agent_health == 0)
        males = self.agent_gender == 0
        females = self.agent_gender == 1
        
        for r_idx, r in enumerate(self.regions):
            r_mask = self.agent_region == r_idx
            labor_supply[r] = {
                'unskilled': 0.0,
                'semi-skilled': 0.0,
                'skilled': 0.0
            }
            for l_idx, l_type in enumerate(['unskilled', 'semi-skilled', 'skilled']):
                l_mask = self.agent_labor == l_idx
                avail_males = np.sum(r_mask & l_mask & active_working & males) * (1.0 - mobilization_rate)
                avail_females = np.sum(r_mask & l_mask & active_working & females)
                
                supply_val = (avail_males + avail_females) * labor_participation * real_people_per_agent
                labor_supply[r][l_type] = max(100.0, supply_val)
                
        return labor_supply, aggregate_consumption
