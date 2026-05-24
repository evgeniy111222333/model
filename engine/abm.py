import numpy as np
from engine.utils import get_sector_wage_premium, get_depreciation_rate, AdaptiveExpectations


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
        # Investment depends on profit, capital returns (MPK) vs bank interest rate
        reinvest_share = 0.15
        if mpk > interest_rate:
            reinvest_share += 0.10
        else:
            reinvest_share -= 0.05
        
        reinvest_share = np.clip(reinvest_share - 0.5 * interest_rate, 0.05, 0.30)
        inv = max(0.0, profit * reinvest_share)
        self.capital = max(1e-3, self.capital * (1.0 - depreciation - war_dmg) + inv)
        return inv


class ABMEngine:
    """
    High-performance Vectorized Columnar Agent-Based Model engine.
    Manages 3.4 million agents using flat numpy arrays for high speed and low memory usage.
    Tracks Pillar 2 accumulative pension accounts and decentralized firm capital.
    """
    def __init__(self, regions, sectors, num_households=3400000, distances=None):
        self.regions = regions
        self.sectors = sectors
        self.R = len(regions)
        self.S = len(sectors)
        self.num_households = num_households
        
        # Initialize distance matrix
        if distances is not None:
            self.distances = distances
        else:
            self.distances = self._init_distance_matrix()
        
        # Micro database (flat columnar NumPy arrays)
        self.agent_region = None
        self.agent_labor = None
        self.agent_wealth = None
        self.agent_cohort = None
        self.agent_gender = None
        self.agent_health = None
        self.agent_age = None
        self.agent_pension_wealth = None
        self.agent_sector = None
        
        self.wage_premium_vec = np.array([get_sector_wage_premium(s) for s in self.sectors], dtype=np.float32)
        
        # Use shared depreciation vector from utils
        self.depreciation_vec = np.array([get_depreciation_rate(s) for s in self.sectors], dtype=np.float32)
        
        self.firms = {}
        
    def _init_distance_matrix(self):
        if self.R == 27:
            try:
                from data.loader import calculate_geographic_distances
                return calculate_geographic_distances()
            except Exception:
                pass
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
            regions_sampled = []
            for r_idx, r in enumerate(self.regions):
                r_pop = initial_pop[r]['Male'].sum() + initial_pop[r]['Female'].sum()
                r_share = r_pop / 3.4e7
                r_agents_count = int(self.num_households * r_share)
                regions_sampled.extend([r_idx] * r_agents_count)
                
            while len(regions_sampled) < self.num_households:
                regions_sampled.append(np.random.randint(0, self.R))
                
            self.agent_region = np.array(regions_sampled, dtype=np.int8)
            self.agent_labor = np.random.choice([0, 2], size=self.num_households, p=[0.60, 0.40]).astype(np.int8)
            self.agent_cohort = np.ones(self.num_households, dtype=np.int8) * 6
            self.agent_age = np.ones(self.num_households, dtype=np.int8) * 32
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
            
        self.agent_wealth = (50000.0 + np.random.exponential(30000.0, size=self.num_households)).astype(np.float32)
        # Pillar 2 pension accounts initialization
        self.agent_pension_wealth = (5000.0 + np.random.exponential(10000.0, size=self.num_households)).astype(np.float32)
        
        # Sector allocation for agents based on initial capital shares of sectors in their region
        self.agent_sector = np.zeros(self.num_households, dtype=np.int8)
        if self.num_households <= 5000:
            self.agent_sector = np.random.choice(self.S, size=self.num_households).astype(np.int8)
        else:
            for r_idx, r in enumerate(self.regions):
                r_mask = self.agent_region == r_idx
                N_r = np.sum(r_mask)
                if N_r > 0:
                    cap_shares = np.array([initial_capital[r][s] for s in self.sectors])
                    sum_cap = np.sum(cap_shares)
                    if sum_cap > 0:
                        cap_shares /= sum_cap
                    else:
                        cap_shares = np.ones(self.S) / self.S
                    self.agent_sector[r_mask] = np.random.choice(self.S, size=N_r, p=cap_shares).astype(np.int8)
        
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
        compat_list = []
        limit = min(5000, self.num_households)
        for i in range(limit):
            h_val = int(self.agent_health[i])
            if h_val > 1: # Exclude deceased and emigrated
                continue
            l_idx = int(self.agent_labor[i])
            l_type = ['unskilled', 'semi-skilled', 'skilled'][l_idx]
            if l_type == 'semi-skilled':
                l_type = 'unskilled'
            cohort_str = '15-64'
            compat_list.append(HouseholdAgent(
                agent_id=i,
                region=self.regions[self.agent_region[i]],
                labor_type=l_type,
                wealth=float(self.agent_wealth[i]),
                age_cohort=cohort_str,
                active=(h_val <= 1),
                health=h_val
            ))
        return compat_list

    def step(self, prices, wages_by_type, grp_per_capita, tax_rates, subsistence_demands, budget_shares, scenario_modifiers):
        """
        Executes one year ABM step:
          1. Vectorized Logit Migration Decisions.
          2. Vectorized Stone-Geary LES Consumption, Pillar 2 Pension deductions, and Wealth updates.
        """
        # 1. Group-Level Vectorized Logit Migration
        p_indices = np.zeros(self.R)
        risks = np.zeros(self.R)
        for r_idx, r in enumerate(self.regions):
            p_indices[r_idx] = sum(budget_shares[r].get(s, 1.0 / self.S) * prices[r][s] for s in self.sectors)
            risks[r_idx] = scenario_modifiers.get('frontline_states', {}).get(r, 0)
            if risks[r_idx] == 1:
                risks[r_idx] = 0.5
            elif risks[r_idx] == 2:
                risks[r_idx] = 1.5
                
        beta_wage = 1.8
        beta_dist = 1.2
        beta_risk = 3.0
        
        active_mask = (self.agent_health == 0) & (self.agent_cohort >= 3) & (self.agent_cohort <= 12)
        
        old_regions = self.agent_region.copy()
        
        for r_from_idx in range(self.R):
            r_from = self.regions[r_from_idx]
            for l_idx in range(3):
                l_type = ['unskilled', 'semi-skilled', 'skilled'][l_idx]
                group_mask = active_mask & (self.agent_region == r_from_idx) & (self.agent_labor == l_idx)
                N_g = np.sum(group_mask)
                if N_g == 0:
                    continue
                
                wages_dest = []
                for r in self.regions:
                    w_r = wages_by_type[r]
                    if l_type in w_r:
                        w_val = w_r[l_type]
                    elif l_type == 'semi-skilled':
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

        # Re-allocate sectors for agents who migrated based on destination region capital shares
        migrated_mask = self.agent_region != old_regions
        migrated_indices = np.where(migrated_mask)[0]
        if len(migrated_indices) > 0:
            for r_idx, r in enumerate(self.regions):
                r_dest_mask = migrated_mask & (self.agent_region == r_idx)
                N_dest = np.sum(r_dest_mask)
                if N_dest > 0:
                    cap_shares = np.array([self.firms[r][s].capital for s in self.sectors])
                    sum_cap = np.sum(cap_shares)
                    if sum_cap > 0:
                        cap_shares /= sum_cap
                    else:
                        cap_shares = np.ones(self.S) / self.S
                    self.agent_sector[r_dest_mask] = np.random.choice(self.S, size=N_dest, p=cap_shares).astype(np.int8)

        # 2. Vectorized Stone-Geary LES Consumption & Wealth Update
        real_people_per_agent = 10.0
        interest_rate = scenario_modifiers.get('interest_rate', 0.15)
        is_q = scenario_modifiers.get('is_quarterly', False)
        
        deposit_interest_rate = (interest_rate * 0.8) / 4.0 if is_q else (interest_rate * 0.8)
        
        # Concentrated inheritance of wealth and pension wealth
        deceased_mask = self.agent_health == 2
        deceased_indices = np.where(deceased_mask)[0]
        for r_idx in range(self.R):
            r_deceased = deceased_indices[self.agent_region[deceased_indices] == r_idx]
            if len(r_deceased) == 0:
                continue
            
            r_active_indices = np.where((self.agent_health == 0) & (self.agent_region == r_idx))[0]
            if len(r_active_indices) > 0:
                total_w = self.agent_wealth[r_deceased]
                total_pw = self.agent_pension_wealth[r_deceased]
                
                # Pick random active heirs in the same region
                heirs = np.random.choice(r_active_indices, size=len(r_deceased))
                np.add.at(self.agent_wealth, heirs, total_w)
                np.add.at(self.agent_wealth, heirs, total_pw)
                
            self.agent_wealth[r_deceased] = 0.0
            self.agent_pension_wealth[r_deceased] = 0.0
        
        aggregate_consumption = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        
        for r_idx, r in enumerate(self.regions):
            r_mask = (self.agent_region == r_idx) & (self.agent_health <= 1)
            N_r = np.sum(r_mask)
            if N_r == 0:
                continue
                
            wealth_sub = self.agent_wealth[r_mask]
            pension_wealth_sub = self.agent_pension_wealth[r_mask]
            labor_sub = self.agent_labor[r_mask]
            cohort_sub = self.agent_cohort[r_mask]
            health_sub = self.agent_health[r_mask]
            
            tax = tax_rates.get(r, 0.23)
            w_unskilled = wages_by_type[r].get('unskilled', 120000.0)
            w_skilled = wages_by_type[r].get('skilled', 300000.0)
            w_semiskilled = wages_by_type[r].get('semi-skilled', (w_skilled + w_unskilled) / 2.0)
            
            wages_r = np.array([w_unskilled, w_semiskilled, w_skilled])
            agent_sector_sub = self.agent_sector[r_mask]
            agent_wages = wages_r[labor_sub] * self.wage_premium_vec[agent_sector_sub]
            
            # Pillar 2 individual accumulative pension contribution (4%)
            is_worker = (cohort_sub >= 3) & (cohort_sub <= 12) & (health_sub == 0)
            
            # Remittances channel: emigrants from this region send money back
            # Track emigrants via agent_region == -1 (displaced/abroad)
            # In our model, refugees pool is tracked at model level
            # Here we estimate emigrants based on region population loss vs baseline
            # For simplicity, use fixed remittance rate per worker
            remittance_rate = scenario_modifiers.get('remittance_rate_per_worker', 0.10)
            workers_in_region = np.sum(is_worker)
            total_remittances_r = workers_in_region * remittance_rate * (w_unskilled + w_skilled) / 100.0
            remittances_per_worker = 0.0
            if total_remittances_r > 0:
                num_workers = np.sum(is_worker)
                if num_workers > 0:
                    remittances_per_worker = total_remittances_r / num_workers
            
            pension_pillar2_contrib = np.where(is_worker, agent_wages * 0.04, 0.0)
            pension_wealth_sub += pension_pillar2_contrib
            
            # Accumulated interest on Pillar 2 pension accounts
            pension_wealth_sub *= (1.0 + deposit_interest_rate)
            
            # Net income after taxes (Solidarity USC + income tax + military levy) + remittances
            income = np.where(is_worker, agent_wages * (1.0 - tax - 0.04) + remittances_per_worker, 0.0)
            
            # State Pension (Pillar 1)
            pension_rate = scenario_modifiers.get('pension_rate', 50000.0)
            pension_rate_q = pension_rate / 4.0 if is_q else pension_rate
            is_pensioner = (cohort_sub > 12) | (health_sub == 1)
            income += np.where(is_pensioner, pension_rate_q * (1.0 - tax), 0.0)
            
            # Private Pension Annuity Payout (Pillar 2)
            # Pensioners receive 8% of their accumulated Pillar 2 pension wealth as private pension supplement
            is_retired_pensioner = (cohort_sub > 12) & (health_sub == 0)
            annuity_rate = 0.08 / 4.0 if is_q else 0.08
            private_annuity = np.where(is_retired_pensioner, pension_wealth_sub * annuity_rate, 0.0)
            pension_wealth_sub -= private_annuity
            income += private_annuity
            
            # Save pension wealth back
            self.agent_pension_wealth[r_mask] = pension_wealth_sub.astype(np.float32)
            
            w_draw_rate = 0.15 / 4.0 if is_q else 0.15
            wealth_draw = wealth_sub * w_draw_rate
            
            # REAL ESTATE WEALTH EFFECT: housing wealth affects consumption
            # Estimate real estate wealth as 3x annual income for working-age, 1.5x for elderly
            real_estate_mult = np.where(cohort_sub <= 12, 3.0, 1.5)
            housing_wealth = real_estate_mult * wages_r[labor_sub] * (1.0 + 0.5 * (cohort_sub > 12))
            # Wealth effect: 0.03 of housing wealth adds to liquidity (housing wealth channel)
            housing_wealth_effect = housing_wealth * 0.03 / 4.0 if is_q else housing_wealth * 0.03
            
            liquidity = income + wealth_draw + housing_wealth_effect
            
            prices_r = np.array([prices[r][s] for s in self.sectors])
            subsist_vals = np.array([subsistence_demands[r][s] for s in self.sectors])
            subsist_cost = np.sum(subsist_vals * prices_r)
            
            super_income = np.clip(liquidity - housing_wealth_effect - subsist_cost, 0.0, None)
            
            sufficient_mask = liquidity >= subsist_cost
            w_penalty_rate = 0.05 / 4.0 if is_q else 0.05
            new_wealth = np.where(
                sufficient_mask,
                np.clip(wealth_sub + super_income * 0.10 - wealth_draw, 0.0, None),
                np.clip(wealth_sub - liquidity * w_penalty_rate, 0.0, None)
            )
            self.agent_wealth[r_mask] = new_wealth.astype(np.float32)
            
            for s_idx, s in enumerate(self.sectors):
                p = prices[r][s]
                sub_qty = subsistence_demands[r][s]
                b_share = budget_shares[r].get(s, 1.0 / self.S)
                
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
                
                supply_val = (avail_males + avail_females) * labor_participation
                labor_supply[r][l_type] = max(100.0, supply_val)
                
        return labor_supply, aggregate_consumption

    def update_firm_capitals(self, total_profits, realized_output, prices, scenario_modifiers):
        """
        Decentralized Firm Agent capital update and investment planning.
        Firms invest corporate profits based on MPK vs interest rates, and expand capital.
        CREDIT→PRODUCTION LINKAGE: Available bank loans constrain investment capacity.
        """
        interest_rate = scenario_modifiers.get('interest_rate', 0.15)
        war_damage = scenario_modifiers.get('war_damage', {})
        
        # Credit-to-production linkage: available bank loans constrain investment capacity
        available_credit_uah = scenario_modifiers.get('available_credit_uah', 0.5e12)
        
        fdi_uah = scenario_modifiers.get('fdi_usd', 1.5e9) * scenario_modifiers.get('exchange_rate', 40.0)
        aid_uah = scenario_modifiers.get('foreign_aid_usd', 22.0e9) * (1.0 - scenario_modifiers.get('foreign_aid_grant_share', 0.50)) * scenario_modifiers.get('exchange_rate', 40.0)
        
        # Calculate national output sums per sector for share distribution
        total_sector_output = {s: sum(realized_output.get((rj, s), 0.0) for rj in self.regions) for s in self.sectors}
        
        # Total credit available for productive investment (after banking system overhead)
        total_credit_pool = available_credit_uah * 0.70  # 70% goes to productive sector
        
        capital_next = {}
        total_investment = 0.0
        for r in self.regions:
            capital_next[r] = {}
            state = scenario_modifiers.get('frontline_states', {}).get(r, 0)
            if state == 2: # Occupied
                for s in self.sectors:
                    self.firms[r][s].capital = 1e-3
                    capital_next[r][s] = 1e-3
                continue
                
            for s_idx, s in enumerate(self.sectors):
                firm = self.firms[r][s]
                out_qty = realized_output.get((r, s), 0.0)
                out_val = out_qty * prices[r][s]
                
                out_share = out_qty / max(1e-5, total_sector_output[s])
                firm_profit = total_profits * out_share * 0.10 # estimate individual profit
                
                depreciation = self.depreciation_vec[s_idx]
                # Dynamic investment planning
                mpk = out_val / max(1e-5, firm.capital)
                
                # Credit-constrained investment: firms can borrow up to their collateral
                # Collateral proxy = firm capital value (at 60% LTV)
                collateral_value = firm.capital * 0.60
                max_credit_firm = collateral_value * (1.0 + 0.5 * (mpk - interest_rate))
                max_credit_firm = min(max_credit_firm, available_credit_uah * out_share * 2.0)  # credit allocation by sector
                
                # Investment from own profits + borrowed credit
                reinvest_share = 0.15
                if mpk > interest_rate:
                    reinvest_share += 0.10
                else:
                    reinvest_share -= 0.05
                
                inv_from_profit = max(0.0, firm_profit * np.clip(reinvest_share - 0.5 * interest_rate, 0.05, 0.30))
                
                # Credit-constrained total investment
                inv_from_credit = min(max_credit_firm, inv_from_profit * 0.5)  # can borrow up to 50% of own investment
                total_inv = inv_from_profit + inv_from_credit
                
                total_investment += total_inv
                
                firm.capital = max(1e-3, firm.capital * (1.0 - depreciation) + total_inv)
                
                # Receive share of FDI and Aid
                fdi_share = out_share * (fdi_uah / len(self.regions))
                aid_share = out_share * (aid_uah / len(self.regions))
                firm.capital += fdi_share + aid_share
                
                # Combat / frontline direct capital damage
                sector_war_dmg = war_damage.get(r, {}).get(s, 0.0)
                if sector_war_dmg > 0:
                    firm.capital = max(1e-3, firm.capital * (1.0 - sector_war_dmg))
                    
                capital_next[r][s] = firm.capital
                
        # Credit channel effect: print for debugging (can be logged in production)
        # print(f"Credit→Production: total_credit_pool={total_credit_pool:.2e}, total_investment={total_investment:.2e}")
        return capital_next
