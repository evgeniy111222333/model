import numpy as np

class HouseholdAgent:
    """
    Individual household agent on the micro-level.
    Handles Stone-Geary utility-maximizing consumption and Logit-based regional migration.
    """
    def __init__(self, agent_id, region, labor_type, wealth, age_cohort):
        self.agent_id = agent_id
        self.region = region
        self.labor_type = labor_type # 'skilled' or 'unskilled'
        self.wealth = wealth # in UAH
        self.age_cohort = age_cohort # '0-14', '15-64', '65+'
        self.active = (age_cohort == '15-64')

    def calculate_consumption(self, prices, wage, tax_rate, subsistence, budget_shares):
        """
        Stone-Geary Linear Expenditure System (LES).
        Minimizes subsistence demands first, then distributes supernumerary income.
        """
        # Income = wage + interest on savings/wealth
        income = (wage if self.active else 0.0) * (1.0 - tax_rate)
        # Use a fraction of wealth as additional liquidity
        liquidity = income + self.wealth * 0.15
        
        # Calculate total cost of subsistence consumption
        subsist_cost = 0.0
        for s, sub_qty in subsistence.items():
            subsist_cost += sub_qty * prices[self.region][s]
            
        realized_consumption = {}
        
        if liquidity >= subsist_cost:
            # Under Stone-Geary: C_s = subsistence_s + (beta_s / P_s) * (Liquidity - Subsistence_Cost)
            super_income = liquidity - subsist_cost
            for s, sub_qty in subsistence.items():
                p = prices[self.region][s]
                share = budget_shares.get(s, 1.0 / len(subsistence))
                qty = sub_qty + (share * super_income) / p
                realized_consumption[s] = qty
            
            # Save remainder to wealth
            self.wealth = max(0.0, self.wealth + (liquidity - subsist_cost) * 0.10)
        else:
            # Ration consumption proportionally to subsistence if liquidity is insufficient
            ratio = liquidity / max(1.0, subsist_cost)
            for s, sub_qty in subsistence.items():
                realized_consumption[s] = sub_qty * ratio
            self.wealth = max(0.0, self.wealth - liquidity * 0.05) # Drawdown savings
            
        return realized_consumption

    def migrate_decision(self, regions, grp_per_capita, wages_by_type, prices, distances, r_idx, scenario_modifiers):
        """
        Utility-maximizing Logit Choice Model for migration:
        V_j = beta_wage * ln(Wage_j / PriceIndex_j) - beta_dist * Distance_ij - beta_risk * Risk_j
        """
        if not self.active:
            # Elderly and children migrate less independently; they follow working cohort trends.
            # For simplicity, we only let working-age agents make migration decisions.
            return self.region

        beta_wage = 1.5
        beta_dist = 0.8
        beta_risk = 2.0
        
        # Compute utilities for all possible destinations
        utilities = []
        for j_idx, r_to in enumerate(regions):
            if r_to == self.region:
                # Add inert utility to stay in the current region
                v = 1.0
            else:
                # Real wage in target region
                w_to = wages_by_type[r_to][self.labor_type]
                # Price index proxy: average price of consumer goods & energy
                p_index = (prices[r_to]['ConsumerGoods'] + prices[r_to]['Energy']) / 2.0
                real_wage = w_to / max(1e-2, p_index)
                
                # Distance penalty
                dist = distances[r_idx, j_idx] / 100.0 # normalized
                
                # Risk penalty (war damage modifier in target region)
                war_dmg = scenario_modifiers.get('war_damage', {}).get(r_to, {}).get('Construction', 0.0)
                
                v = beta_wage * np.log(max(1e-1, real_wage)) - beta_dist * dist - beta_risk * war_dmg
                
            utilities.append(v)
            
        # Logit probabilities
        utilities = np.array(utilities)
        # Shift to avoid numerical overflow in exp
        utilities -= np.max(utilities)
        probs = np.exp(utilities) / np.sum(np.exp(utilities))
        
        # Sample next region
        new_region = np.random.choice(regions, p=probs)
        self.region = new_region
        return new_region


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
        """
        Updates capital stock based on firm investment budgeting:
        K(t+1) = (1 - delta - dmg) * K(t) + I(t)
        """
        # Reinvestment is driven by profit margins relative to interest rates
        reinvest_share = 0.15
        if mpk > interest_rate:
            reinvest_share += 0.10 # Boost investment if capital is highly productive
            
        inv = max(0.0, profit * reinvest_share)
        self.capital = max(1e-3, self.capital * (1.0 - depreciation - war_dmg) + inv)
        return inv


class ABMEngine:
    """
    Manages the micro-level simulation of household and firm populations.
    """
    def __init__(self, regions, sectors, num_households=5000):
        self.regions = regions
        self.sectors = sectors
        self.num_households = num_households
        
        # Initialize distance matrix
        self.distances = self._init_distance_matrix()
        
        # Will be populated during simulation steps
        self.households = []
        self.firms = {}
        
    def _init_distance_matrix(self):
        n = len(self.regions)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist[i, j] = 0.0
                else:
                    dist[i, j] = 100.0 + abs(i - j) * 35.0
        return dist

    def initialize_agents(self, initial_pop, initial_capital):
        """
        Creates agent objects matching the aggregate population and capital distributions.
        """
        np.random.seed(42)
        self.households = []
        
        # Distribute household agents proportionally to regional demographics
        total_pop_sum = sum(initial_pop[r]['Male'].sum() + initial_pop[r]['Female'].sum() for r in self.regions)
        
        agent_id = 0
        for r in self.regions:
            r_pop = initial_pop[r]['Male'].sum() + initial_pop[r]['Female'].sum()
            r_share = r_pop / total_pop_sum
            r_agents_count = int(self.num_households * r_share)
            
            # Split by labor type: 40% skilled, 60% unskilled
            for _ in range(r_agents_count):
                lab_type = 'skilled' if np.random.rand() < 0.40 else 'unskilled'
                cohort = '15-64' # Focus primarily on active labor forces for ABM migration
                wealth = 50000.0 + np.random.exponential(30000.0) # Initial UAH savings
                
                self.households.append(HouseholdAgent(
                    agent_id=agent_id,
                    region=r,
                    labor_type=lab_type,
                    wealth=wealth,
                    age_cohort=cohort
                ))
                agent_id += 1
                
        # Initialize firms (1 firm per region-sector)
        self.firms = {}
        for r in self.regions:
            self.firms[r] = {}
            for s in self.sectors:
                self.firms[r][s] = FirmAgent(
                    region=r,
                    sector=s,
                    initial_capital=initial_capital[r][s]
                )

    def step(self, prices, wages_by_type, grp_per_capita, tax_rates, subsistence_demands, budget_shares, scenario_modifiers):
        """
        Steps the micro agents and returns:
            - Labor supply by type per region
            - Aggregate consumption demand by region-sector
            - Aggregate investment by region-sector
        """
        # 1. Migrate Households
        for agent in self.households:
            r_idx = self.regions.index(agent.region)
            agent.migrate_decision(
                regions=self.regions,
                grp_per_capita=grp_per_capita,
                wages_by_type=wages_by_type,
                prices=prices,
                distances=self.distances,
                r_idx=r_idx,
                scenario_modifiers=scenario_modifiers
            )
            
        # 2. Calculate labor supply counts per region
        labor_supply = {r: {'skilled': 0.0, 'unskilled': 0.0} for r in self.regions}
        # Scale factor (how many real people 1 agent represents)
        real_people_per_agent = 15.0e6 / len(self.households)
        
        # Adjust labor supply for mobilization rates (mostly unskilled or skilled males)
        mob_rate = scenario_modifiers.get('mobilization_rate', 0.02)
        
        for agent in self.households:
            if agent.active:
                # Subtract mobilized fraction
                supply_val = real_people_per_agent * (1.0 - mob_rate)
                labor_supply[agent.region][agent.labor_type] += supply_val
                
        # 3. Calculate household consumption demands
        aggregate_consumption = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
        for agent in self.households:
            # Determine applicable wage
            w = wages_by_type[agent.region][agent.labor_type]
            tax = tax_rates.get(agent.region, 0.23) # Pit + Military
            
            c_dict = agent.calculate_consumption(
                prices=prices,
                wage=w,
                tax_rate=tax,
                subsistence=subsistence_demands[agent.region],
                budget_shares=budget_shares
            )
            
            # Aggregate consumption scale
            for s, qty in c_dict.items():
                aggregate_consumption[agent.region][s] += qty * real_people_per_agent

        return labor_supply, aggregate_consumption
