import numpy as np
import copy
from engine.demographics import DemographicEngine
from engine.production import ProductionEngine
from engine.abm import ABMEngine
from engine.cge import CGESolver
from engine.finance import FinanceEngine

class ModelRunner:
    """
    Main controller for the Hybrid AB-CGE Ukraine Economic Simulator.
    Coordinates demographic Leslie aging, micro agent-based migration and consumption,
    macro general equilibrium clearing, and commercial banking/fiscal policy.
    Supports 93 sub-sectors and dynamic frontline transitions.
    """
    def __init__(self, base_data, num_households=3400000):
        self.regions = base_data['regions']
        self.sectors = base_data['sectors']
        self.R = len(self.regions)
        self.S = len(self.sectors)
        self.initial_pop = copy.deepcopy(base_data['initial_pop'])
        
        # Initialize distance and frontline states
        self.distances = base_data.get('distances')
        
        # Initialize Demographics
        self.demographics = DemographicEngine(
            regions=self.regions,
            initial_pop=self.initial_pop,
            fertility_rates=copy.deepcopy(base_data['fertility_rates']),
            mortality_rates=copy.deepcopy(base_data['mortality_rates']),
            migration_gravity_coeffs=copy.deepcopy(base_data['migration_gravity']),
            distances=self.distances
        )
        
        # Initialize Production
        self.production = ProductionEngine(
            regions=self.regions,
            sectors=self.sectors
        )
        
        self.frontline_states = {}
        for r in self.regions:
            if r in ['Donetsk', 'Luhansk', 'Crimea', 'Sevastopol']:
                self.frontline_states[r] = 2 # Occupied
            elif r in ['Kharkiv', 'Zaporizhzhia', 'Kherson', 'Mykolaiv']:
                self.frontline_states[r] = 1 # Frontline
            else:
                self.frontline_states[r] = 0 # Safe
        
        # Initialize ABM Engine & Agents
        self.num_households = num_households
        self.abm = ABMEngine(self.regions, self.sectors, num_households=self.num_households, distances=self.distances)
        self.capital = copy.deepcopy(base_data['initial_capital'])
        self.abm.initialize_agents(self.initial_pop, self.capital)
        
        # Initialize CGE Solver
        self.base_tech = copy.deepcopy(base_data['base_tech_coefficients'])
        self.cge = CGESolver(self.regions, self.sectors, self.base_tech, distances=self.distances)
        
        self.finance = FinanceEngine()
        
        # Core State Variables
        self.tfp = copy.deepcopy(base_data['initial_tfp'])
        self.prices = copy.deepcopy(base_data['prices'])
        self.energy_utilization = copy.deepcopy(base_data['energy_utilization'])
        
        self.budget_shares = copy.deepcopy(base_data['budget_shares'])
        self.subsistence_demands = copy.deepcopy(base_data['subsistence_demands'])
        self.wages_by_type = copy.deepcopy(base_data['wages_by_type'])
        
        self.refugee_pool = 5.0e6

    def _update_frontline_states(self, scenario_name, year):
        """
        Applies dynamic geopolitical trajectories to frontline states based on scenario and year.
        STOCHASTIC FRONTIER: transition events have probability-based timing with uncertainty.
        """
        import numpy as np
        
        # Stochastic transition probability (0.8 = 80% chance per year near target year)
        stochastic_prob = 0.80
        
        if scenario_name == 'optimistic':
            # Rapid liberation with stochastic timing
            if year >= 2027 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Mykolaiv', 0) > 0:
                    self.frontline_states['Mykolaiv'] = max(0, self.frontline_states['Mykolaiv'] - 1)
                if self.frontline_states.get('Kherson', 0) > 0:
                    self.frontline_states['Kherson'] = max(0, self.frontline_states['Kherson'] - 1)
            
            if year >= 2029 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Kharkiv', 0) > 0:
                    self.frontline_states['Kharkiv'] = max(0, self.frontline_states['Kharkiv'] - 1)
                if self.frontline_states.get('Zaporizhzhia', 0) > 0:
                    self.frontline_states['Zaporizhzhia'] = max(0, self.frontline_states['Zaporizhzhia'] - 1)
            
            if year >= 2031 and np.random.rand() < stochastic_prob:
                # Contested transitions (Occupied -> Frontline combat)
                if self.frontline_states.get('Donetsk', 2) == 2 and np.random.rand() < 0.5:
                    self.frontline_states['Donetsk'] = 1
                if self.frontline_states.get('Luhansk', 2) == 2 and np.random.rand() < 0.5:
                    self.frontline_states['Luhansk'] = 1
            
            if year >= 2034 and np.random.rand() < stochastic_prob:
                # Fully liberated
                if self.frontline_states.get('Donetsk', 0) > 0:
                    self.frontline_states['Donetsk'] = max(0, self.frontline_states['Donetsk'] - 1)
                if self.frontline_states.get('Luhansk', 0) > 0:
                    self.frontline_states['Luhansk'] = max(0, self.frontline_states['Luhansk'] - 1)
            
            if year >= 2037 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Crimea', 2) == 2:
                    self.frontline_states['Crimea'] = 1
                if self.frontline_states.get('Sevastopol', 2) == 2:
                    self.frontline_states['Sevastopol'] = 1
            
            if year >= 2041 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Crimea', 0) > 0:
                    self.frontline_states['Crimea'] = max(0, self.frontline_states['Crimea'] - 1)
                if self.frontline_states.get('Sevastopol', 0) > 0:
                    self.frontline_states['Sevastopol'] = max(0, self.frontline_states['Sevastopol'] - 1)
                    
        elif scenario_name == 'pessimistic':
            # Prolonged war of attrition, stochastic border flare-ups
            if 2027 <= year <= 2036:
                # Random fluctuations in border regions
                flareup_prob = 0.15  # 15% chance per year of flare-up
                for r in ['Sumy', 'Chernihiv']:
                    current_state = self.frontline_states.get(r, 0)
                    if np.random.rand() < flareup_prob:
                        self.frontline_states[r] = 1 if current_state == 0 else current_state
                    elif np.random.rand() < flareup_prob * 0.5:
                        self.frontline_states[r] = 0 if current_state == 1 else current_state
            else:
                self.frontline_states['Sumy'] = 0
                self.frontline_states['Chernihiv'] = 0
                
        else: # baseline
            # Stalemate followed by slow long-term stabilization with stochastic timing
            if year >= 2029 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Mykolaiv', 0) > 0:
                    self.frontline_states['Mykolaiv'] = max(0, self.frontline_states['Mykolaiv'] - 1)
            
            if year >= 2031 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Kherson', 0) > 0:
                    self.frontline_states['Kherson'] = max(0, self.frontline_states['Kherson'] - 1)
            
            if year >= 2039 and np.random.rand() < stochastic_prob:
                if self.frontline_states.get('Kharkiv', 0) > 0:
                    self.frontline_states['Kharkiv'] = max(0, self.frontline_states['Kharkiv'] - 1)
                if self.frontline_states.get('Zaporizhzhia', 0) > 0:
                    self.frontline_states['Zaporizhzhia'] = max(0, self.frontline_states['Zaporizhzhia'] - 1)

    # Seasonal demand multipliers [Winter, Spring, Summer, Autumn]
    SEASONAL_FACTORS = {
        # Energy: high in winter, low in summer
        'EnergyThermal':     [1.6, 0.9, 0.6, 0.9],
        'GasHeatSupply':     [1.7, 0.8, 0.5, 0.9],
        'EnergyNuclearGen':  [1.3, 1.0, 0.9, 1.0],
        'EnergyTransmission':[1.3, 0.9, 0.7, 1.0],
        # Agriculture: high autumn harvest, low winter
        'AgriGrain':         [0.5, 1.0, 1.1, 1.4],
        'AgriTechnical':     [0.5, 1.0, 1.1, 1.4],
        'AgriLivestock':     [0.8, 1.0, 1.1, 1.1],
        'FoodProcessing':    [0.9, 1.0, 0.9, 1.2],
        # Construction: high summer, low winter
        'ConstResidential':  [0.4, 1.1, 1.6, 1.0],
        'ConstCommercial':   [0.4, 1.1, 1.6, 0.9],
        'ConstInfrastructure':[0.5, 1.1, 1.5, 0.9],
        'ConstReconstruction':[0.5, 1.0, 1.5, 1.0],
        # Tourism: high summer
        'HotelsTourism':     [0.6, 0.9, 1.8, 0.8],
        'FoodServices':      [0.8, 1.0, 1.3, 0.9],
    }
    
    # CBAM-affected heavy sectors
    CBAM_SECTORS = ['SteelIron', 'MetalProducts', 'ChemicalFertilizers', 'IndustrialChemicals', 'BuildingMaterials']
    
    # Calvo stickiness by sector group
    CALVO_THETA = {
        'rigid': 0.80,    # Regulated energy, utilities, public services
        'sticky': 0.30,   # Machinery, metallurgy, construction
        'flexible': 0.10, # Agriculture, food, retail, IT services
    }
    
    def _get_calvo_theta(s):
        if s in ['EnergyThermal', 'EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste', 
                 'EnergyTransmission', 'GasHeatSupply', 'UtilityServices', 'GasHeatSupply',
                 'PublicAdmin', 'MilitaryDefense', 'LawEnforcement']:
            return ModelRunner.CALVO_THETA['rigid']
        elif s in ['AgriGrain', 'AgriTechnical', 'AgriLivestock', 'FoodProcessing', 'Fishery',
                   'TradeRetail', 'TradeWholesale', 'ITServicesExport', 'ITProductSaaS']:
            return ModelRunner.CALVO_THETA['flexible']
        else:
            return ModelRunner.CALVO_THETA['sticky']

    def run_simulation(self, scenario_name, scenario_engine, num_years=25, lhs_sample=None):
        """
        Runs the Hybrid AB-CGE simulation with 4-quarter seasonal sub-steps.
        Incorporates shadow economy tax scaling, Calvo sticky prices,
        government sector spending allocation, CBAM trade adjustments, and credit-production linkage.
        """
        history = []
        
        for step_idx in range(num_years):
            year = 2026 + step_idx
            
            # 1. Fetch Scenario & Shock Modifiers
            base_mods = scenario_engine.get_deterministic_modifiers(scenario_name, year)
            base_mods['refugee_pool'] = self.refugee_pool
            base_mods['pension_rate'] = self.finance.pension_rate
            
            grp_per_capita = {}
            for r in self.regions:
                pop_total = sum(self.demographics.pop[r][g].sum() for g in ['Male', 'Female'])
                cap_total = sum(self.capital[r].values())
                grp_per_capita[r] = cap_total / max(1e-5, pop_total)
            base_mods['grp_per_capita'] = grp_per_capita
            
            if lhs_sample is not None:
                mods = scenario_engine.apply_stochastic_shocks(base_mods, lhs_sample)
            else:
                mods = base_mods
                mods['war_damage'] = {}
                mods['export_shock'] = 1.0
                
            self._update_frontline_states(scenario_name, year)
            mods['frontline_states'] = copy.deepcopy(self.frontline_states)
            
            # 2. Demographics (annual step)
            demo_results = self.demographics.step(year, mods, abm=self.abm)
            self.refugee_pool = demo_results['refugees_remaining']
            
            # 3. Frontline Displacement
            occupied_indices = [self.regions.index(r) for r in self.regions if self.frontline_states[r] == 2]
            safe_indices = [self.regions.index(r) for r in self.regions if self.frontline_states[r] == 0]
            if len(safe_indices) == 0:
                safe_indices = [r_idx for r_idx in range(self.R) if r_idx not in occupied_indices]
            if len(occupied_indices) > 0:
                in_occupied = np.isin(self.abm.agent_region, occupied_indices) & (self.abm.agent_health != 2)
                if np.sum(in_occupied) > 0:
                    self.abm.agent_region[in_occupied] = np.random.choice(safe_indices, size=np.sum(in_occupied))

            # 4. Energy Grid Constraints (annual + cascading topological failure)
            # Energy zone definitions: West=0-5, Center=6-15, East/South=16-26
            energy_zone_damage = {'West': 0.0, 'Center': 0.0, 'East': 0.0}
            for r_idx, r in enumerate(self.regions):
                state = self.frontline_states[r]
                for s in self.sectors:
                    rec = mods.get('energy_recovery', 0.04)
                    dmg = mods['war_damage'].get(r, {}).get(s, 0.0)
                    curr_util = self.energy_utilization[r][s]
                    next_util = np.clip(curr_util + rec - dmg, 0.20, 1.0)
                    if state == 1:
                        next_util = min(0.50, next_util)
                    elif state == 2:
                        next_util = 0.0
                    self.energy_utilization[r][s] = next_util
                if r_idx >= 16:
                    energy_zone_damage['East'] += sum(mods['war_damage'].get(r, {}).values())
                elif r_idx >= 6:
                    energy_zone_damage['Center'] += sum(mods['war_damage'].get(r, {}).values())
            
            # Cascading failure: heavy East damage spills into Center
            zone_cascade = energy_zone_damage['East'] > 0.5
            if zone_cascade:
                for r_idx, r in enumerate(self.regions):
                    if 6 <= r_idx <= 15:
                        for s in self.sectors:
                            self.energy_utilization[r][s] = max(0.20, self.energy_utilization[r][s] * 0.85)
            
            # 5. Human Capital / TFP
            regional_tfp = copy.deepcopy(self.tfp)
            for r_idx, r in enumerate(self.regions):
                state = self.frontline_states[r]
                r_pop = self.demographics.pop_array[r_idx]
                total_p = np.sum(r_pop)
                disabled_rate = (np.sum(r_pop[:, :, 1]) / total_p) if total_p > 0 else 0.0
                r_mask = (self.abm.agent_region == r_idx) & (self.abm.agent_health != 2)
                total_agents = np.sum(r_mask)
                skilled_rate = (np.sum(r_mask & (self.abm.agent_labor == 2)) / total_agents) if total_agents > 0 else 0.15
                hci = (1.0 - 0.20 * disabled_rate) * (1.0 + 0.15 * skilled_rate)
                frontline_mult = 1.0 if state == 0 else (0.60 if state == 1 else 0.00)
                for s in self.sectors:
                    regional_tfp[r][s] *= hci * frontline_mult

            # EU Integration: progressive export share boost + CBAM tariff cost
            eu_progress = min(1.0, max(0.0, (year - 2026) / 24.0))
            cbam_penalty = eu_progress * 0.15  # up to 15% tariff on CBAM sectors by 2050
            p_world_export = mods.get('p_world_export', np.ones(self.S))
            if p_world_export is None:
                p_world_export = np.ones(self.S)
            else:
                p_world_export = np.array(p_world_export) if not isinstance(p_world_export, np.ndarray) else p_world_export
            for s_idx, s in enumerate(self.sectors):
                if s in ModelRunner.CBAM_SECTORS:
                    p_world_export[s_idx] *= (1.0 - cbam_penalty)
                else:
                    p_world_export[s_idx] *= (1.0 + eu_progress * 0.08)  # EU market access premium
            mods['p_world_export'] = p_world_export

            # Government sector demand allocation (reconstruction + military)
            defense_nominal = mods.get('defense_spending_ratio', 0.25) * sum(self.capital[r].get(s, 0) for r in self.regions for s in self.sectors) * 0.01
            recon_nominal = mods.get('reconstruction_needs_usd', 15.0e9) * self.finance.exchange_rate * mods.get('foreign_aid_grant_share', 0.50) * 0.30
            corruption_leakage = 0.15
            recon_effective = recon_nominal * (1.0 - corruption_leakage)
            
            mil_demand_shares = {
                'MilSmallArms': 0.05, 'MilArmoredVehicles': 0.15, 'MilArtillery': 0.20,
                'MilMissiles': 0.15, 'MilUAVs': 0.20, 'MilEW': 0.15, 'MilNaval': 0.05,
                'MilProtectiveGear': 0.05
            }
            recon_demand_shares = {'ConstReconstruction': 0.60, 'ConstInfrastructure': 0.40}
            
            # ---- 4-QUARTER SUB-LOOP ----
            quarter_labels = ['Winter', 'Spring', 'Summer', 'Autumn']
            annual_labor_supply = {r: {'unskilled': 0.0, 'semi-skilled': 0.0, 'skilled': 0.0} for r in self.regions}
            annual_consumption  = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
            annual_output       = {(r, s): 0.0 for r in self.regions for s in self.sectors}
            annual_imports      = {(r, s): 0.0 for r in self.regions for s in self.sectors}
            annual_exports      = {(r, s): 0.0 for r in self.regions for s in self.sectors}
            
            # Track quarterly solved prices for annual average (corrects price evolution bias)
            annual_price_solved = {r: {s: 0.0 for s in self.sectors} for r in self.regions}
            
            for q_idx, q_label in enumerate(quarter_labels):
                q_mods = copy.deepcopy(mods)
                q_mods['is_quarterly'] = True
                
                # Seasonal subsistence demand scaling
                q_subsistence = {}
                for r in self.regions:
                    q_subsistence[r] = {}
                    for s in self.sectors:
                        factor = ModelRunner.SEASONAL_FACTORS.get(s, [1.0, 1.0, 1.0, 1.0])[q_idx]
                        q_subsistence[r][s] = self.subsistence_demands[r][s] * factor
                
                # ABM quarterly step
                q_labor, q_consumption = self.abm.step(
                    prices=self.prices,
                    wages_by_type=self.wages_by_type,
                    grp_per_capita=grp_per_capita,
                    tax_rates={r: self.finance.tax_pit + self.finance.tax_military for r in self.regions},
                    subsistence_demands=q_subsistence,
                    budget_shares=self.budget_shares,
                    scenario_modifiers=q_mods
                )
                
                # Add government demand allocation to household consumption
                for r_idx, r in enumerate(self.regions):
                    state = self.frontline_states[r]
                    # Military demand — distributed to frontline + safe regions
                    for ms, ms_share in mil_demand_shares.items():
                        if ms in self.sectors:
                            q_consumption[r][ms] = q_consumption[r].get(ms, 0.0) + (defense_nominal * ms_share / (self.R * 4.0))
                    # Reconstruction demand — concentrate on frontline & recently liberated regions
                    recon_mult = 3.0 if state == 1 else (2.0 if r in ['Donetsk', 'Luhansk', 'Kherson', 'Zaporizhzhia'] else 0.5)
                    for rds, rds_share in recon_demand_shares.items():
                        if rds in self.sectors:
                            q_consumption[r][rds] = q_consumption[r].get(rds, 0.0) + (recon_effective * rds_share * recon_mult / (self.R * 4.0))
                
                # CGE quarterly clearing
                q_output, q_prices_solved, q_wages_solved = self.cge.solve_equilibrium(
                    capital=self.capital,
                    labor_supply_by_type=q_labor,
                    tfp=regional_tfp,
                    prices_init=self.prices,
                    energy_utilization=self.energy_utilization,
                    household_demands=q_consumption,
                    exchange_rate=self.finance.exchange_rate,
                    interest_rate=self.finance.interest_rate,
                    p_world_import=q_mods.get('p_world_import', None),
                    p_world_export=p_world_export
                )
                
                # Calvo sticky prices: blend solved equilibrium with previous prices
                for r in self.regions:
                    for s in self.sectors:
                        theta = ModelRunner._get_calvo_theta(s)
                        q_prices_solved[r][s] = (1.0 - theta) * q_prices_solved[r][s] + theta * self.prices[r][s]
                
                # Update running prices gradually (tatonnement over quarters)
                blend = 0.25
                for r in self.regions:
                    for s in self.sectors:
                        self.prices[r][s] = (1.0 - blend) * self.prices[r][s] + blend * q_prices_solved[r][s]
                    for l_type in ['unskilled', 'semi-skilled', 'skilled']:
                        if l_type in q_wages_solved[r]:
                            self.wages_by_type[r][l_type] = (1.0 - blend) * self.wages_by_type[r].get(l_type, 0.0) + blend * q_wages_solved[r][l_type]
                
                # Accumulate annual totals
                for r in self.regions:
                    for l_type in ['unskilled', 'semi-skilled', 'skilled']:
                        annual_labor_supply[r][l_type] += q_labor[r].get(l_type, 0.0) * 0.25
                    for s in self.sectors:
                        annual_consumption[r][s] += q_consumption[r].get(s, 0.0) * 0.25
                for r in self.regions:
                    for s in self.sectors:
                        annual_output[(r, s)] += q_output.get((r, s), 0.0) * 0.25
                        annual_imports[(r, s)] += self.cge.realized_imports.get((r, s), 0.0) * 0.25
                        annual_exports[(r, s)] += self.cge.realized_exports.get((r, s), 0.0) * 0.25
                        # Accumulate solved prices (average later for annual GDP calculation)
                        annual_price_solved[r][s] += q_prices_solved[r][s] * 0.25
            
            # Sync final quarter prices / wages back
            realized_output = annual_output
            labor_supply = annual_labor_supply
            aggregate_consumption = annual_consumption
            self.cge.realized_imports = annual_imports
            self.cge.realized_exports = annual_exports
            
# 8. Financial aggregates (annual sums)
            # Use average quarterly prices for annual nominal GDP (corrects price evolution bias)
            annual_avg_prices = {r: {s: annual_price_solved[r][s] for s in self.sectors} for r in self.regions}
            
            total_wages = 0.0
            total_profits = 0.0
            regional_grp_nominal = {r: 0.0 for r in self.regions}
            regional_grp_real = {r: 0.0 for r in self.regions}
            
            for r in self.regions:
                state = self.frontline_states[r]
                if state == 2:
                    continue
                ws = self.wages_by_type[r]['skilled']
                wm = self.wages_by_type[r].get('semi-skilled', ws * 0.70)
                wu = self.wages_by_type[r]['unskilled']
                ls = labor_supply[r]['skilled']
                lm = labor_supply[r].get('semi-skilled', ls * 0.70)
                lu = labor_supply[r]['unskilled']
                lab_cost = (ls * ws) + (lm * wm) + (lu * wu)
                total_wages += lab_cost
                
                r_dest_idx = self.regions.index(r)
                for s_idx, s in enumerate(self.sectors):
                    # Use AVERAGE quarterly price for annual nominal output
                    out_qty = annual_output[(r, s)]
                    out_val_nominal = out_qty * annual_avg_prices[r][s]
                    
                    # Intermediate cost: use technology coefficients B_mat with local prices
                    # DO NOT use trade_shares - inter-regional trade is already embedded in CGE output
                    int_cost_nominal = 0.0
                    for sx_idx, sx in enumerate(self.sectors):
                        coeff = self.base_tech.get(s, {}).get(sx, 0.0)
                        if coeff > 0:
                            # Get quantity of input needed
                            qty_needed = out_qty * coeff
                            # Use LOCAL source region price (no trade share distortion)
                            int_cost_nominal += qty_needed * annual_avg_prices[r][sx]
                    
                    # GRP = sum of sectoral gross value added (output - intermediate costs)
                    # This IS the regionalGRP - not double-counting, not summing output
                    grp_sector = out_val_nominal - int_cost_nominal
                    # Ensure GRP is non-negative (production function guarantees this if correct)
                    regional_grp_nominal[r] += max(0.0, grp_sector)
                    out_val_real = out_qty
                    int_cost_real = sum(out_qty * self.base_tech.get(s, {}).get(sx, 0.0) for sx in self.sectors)
                    grp_real_sector = out_val_real - int_cost_real
                    regional_grp_real[r] += max(0.0, grp_real_sector)
            
            real_gdp_uah = sum(regional_grp_real.values())
            nominal_gdp_uah = sum(regional_grp_nominal.values())
            nominal_gdp_usd = nominal_gdp_uah / self.finance.exchange_rate
            
            total_exports_usd = sum(annual_exports.get((r, s), 0.0) for r in self.regions for s in self.sectors if self.frontline_states[r] != 2) / self.finance.exchange_rate
            total_imports_usd = sum(annual_imports.get((r, s), 0.0) for r in self.regions for s in self.sectors if self.frontline_states[r] != 2) / self.finance.exchange_rate
            
            household_wealth_sum = float(np.sum(self.abm.agent_wealth[self.abm.agent_health != 2]))
            
            if self.abm is not None:
                residing = self.abm.agent_health <= 1
                num_pensioners = float(np.sum(residing & ((self.abm.agent_cohort > 12) | (self.abm.agent_health == 1))))
            else:
                num_pensioners = float(np.sum(self.demographics.pop_array[:, 13:, :, :]) + np.sum(self.demographics.pop_array[:, :13, :, 1]))
            
            # Shadow economy: 30% of wages/profits are informal and avoid tax
            # FinanceEngine sees only 70% of taxable wages/profits
            shadow_economy_rate = mods.get('shadow_economy_rate', 0.30)
            taxable_wages = total_wages * (1.0 - shadow_economy_rate)
            taxable_profits = total_profits * (1.0 - shadow_economy_rate)
            
            fin_results = self.finance.step(
                year=year,
                nominal_gdp_uah=nominal_gdp_uah,
                nominal_gdp_usd=nominal_gdp_usd,
                total_wages=taxable_wages,
                corporate_profits=taxable_profits,
                exports_usd=total_exports_usd,
                imports_usd=total_imports_usd,
                scenario_modifiers=mods,
                household_wealth_sum=household_wealth_sum,
                num_pensioners=num_pensioners
            )
            
            # Credit-to-production linkage: available bank loans constrain investment capacity
            mods['interest_rate'] = fin_results['interest_rate']
            mods['exchange_rate'] = fin_results['exchange_rate']
            available_credit_uah = fin_results['bank_loans']
            mods['available_credit_uah'] = available_credit_uah
            
            # 9. Decentralized Firm Capital Accumulation
            self.capital = self.abm.update_firm_capitals(
                total_profits=total_profits,
                realized_output=realized_output,
                prices=self.prices,
                scenario_modifiers=mods
            )
            
            # TFP growth step
            tfp_growth_by_region = mods.get('tfp_growth_by_region', {})
            for r in self.regions:
                growth_rate = tfp_growth_by_region.get(r, mods.get('tfp_growth', 0.018))
                for s in self.sectors:
                    self.tfp[r][s] *= (1.0 + growth_rate)
            
            snapshot = {
                'year': year,
                'gdp_real_uah': real_gdp_uah,
                'gdp_nominal_usd': nominal_gdp_usd,
                'population': demo_results['total_pop'],
                'births': demo_results['births'],
                'deaths': demo_results['deaths'],
                'refugees_remaining': demo_results['refugees_remaining'],
                'inflation': fin_results['inflation_rate'],
                'exchange_rate': fin_results['exchange_rate'],
                'interest_rate': fin_results['interest_rate'],
                'debt_gdp': fin_results['debt_gdp'],
                'deficit_gdp': fin_results['deficit_gdp'],
                'trade_balance_usd': fin_results['trade_balance_usd'],
                'tax_revenue_uah': fin_results['tax_revenue_uah'],
                'bank_loans': fin_results['bank_loans'],
                'bank_deposits': fin_results['bank_deposits'],
                'nbu_fx_reserves_usd': fin_results.get('nbu_fx_reserves_usd', 38.0e9),
                'pension_rate': fin_results.get('pension_rate', 50000.0),
                'deposits_pension_pillar2': fin_results.get('deposits_pension_pillar2', 0.0),
                'insurance_equity': fin_results.get('insurance_equity', 0.0),
                'insurance_reserves': fin_results.get('insurance_reserves', 0.0),
                'shadow_economy_gdp_share': shadow_economy_rate,
                'energy_zone_cascade': zone_cascade,
                'eu_integration_progress': eu_progress,
                'cbam_penalty': cbam_penalty,
                'regional_data': {r: {
                    'grp_real': regional_grp_real[r],
                    'grp_nominal': regional_grp_nominal[r],
                    'pop': sum(self.demographics.pop[r][g].sum() for g in ['Male', 'Female']),
                    'wage_skilled': self.wages_by_type[r]['skilled'],
                    'wage_unskilled': self.wages_by_type[r]['unskilled'],
                    'frontline_state': int(self.frontline_states[r]),
                    'tfp_average': float(sum(self.tfp[r].values()) / len(self.tfp[r])),
                    'sectors': {s: {
                        'output': realized_output[(r, s)],
                        'capital': self.capital[r][s],
                        'price': self.prices[r][s]
                    } for s in self.sectors}
                } for r in self.regions}
            }
            history.append(snapshot)
            
        return history
