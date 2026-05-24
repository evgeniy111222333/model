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
        """
        if scenario_name == 'optimistic':
            # Rapid liberation
            if year == 2028:
                self.frontline_states['Mykolaiv'] = 0
                self.frontline_states['Kherson'] = 0
            elif year == 2030:
                self.frontline_states['Kharkiv'] = 0
                self.frontline_states['Zaporizhzhia'] = 0
            elif year == 2032:
                # Contested transitions (Occupied -> Frontline combat)
                self.frontline_states['Donetsk'] = 1
                self.frontline_states['Luhansk'] = 1
            elif year == 2035:
                # Fully liberated
                self.frontline_states['Donetsk'] = 0
                self.frontline_states['Luhansk'] = 0
            elif year == 2038:
                self.frontline_states['Crimea'] = 1
                self.frontline_states['Sevastopol'] = 1
            elif year == 2042:
                self.frontline_states['Crimea'] = 0
                self.frontline_states['Sevastopol'] = 0
        elif scenario_name == 'pessimistic':
            # Prolonged war of attrition, border flare-ups
            if 2028 <= year <= 2035:
                self.frontline_states['Sumy'] = 1
                self.frontline_states['Chernihiv'] = 1
            else:
                self.frontline_states['Sumy'] = 0
                self.frontline_states['Chernihiv'] = 0
        else: # baseline
            # Stalemate followed by slow long-term stabilization
            if year == 2030:
                self.frontline_states['Mykolaiv'] = 0
            elif year == 2032:
                self.frontline_states['Kherson'] = 0
            elif year == 2040:
                self.frontline_states['Kharkiv'] = 0
                self.frontline_states['Zaporizhzhia'] = 0

    def run_simulation(self, scenario_name, scenario_engine, num_years=25, lhs_sample=None):
        """
        Runs the Hybrid AB-CGE simulation.
        """
        history = []
        
        for step_idx in range(num_years):
            year = 2026 + step_idx
            
            # 1. Fetch Scenario & Shock Modifiers
            base_mods = scenario_engine.get_deterministic_modifiers(scenario_name, year)
            base_mods['refugee_pool'] = self.refugee_pool
            base_mods['pension_rate'] = self.finance.pension_rate
            
            # GRP per capita for demographics gravity
            grp_per_capita = {}
            for r in self.regions:
                pop_total = sum(self.demographics.pop[r][g].sum() for g in ['Male', 'Female'])
                cap_total = sum(self.capital[r].values())
                grp_per_capita[r] = cap_total / max(1e-5, pop_total)
            base_mods['grp_per_capita'] = grp_per_capita
            
            # Apply Monte Carlo LHS stochastic shocks
            if lhs_sample is not None:
                mods = scenario_engine.apply_stochastic_shocks(base_mods, lhs_sample)
            else:
                mods = base_mods
                mods['war_damage'] = {}
                mods['export_shock'] = 1.0
                
            # Update frontline states dynamically
            self._update_frontline_states(scenario_name, year)
            mods['frontline_states'] = copy.deepcopy(self.frontline_states)
            
            # 2. Demographic Step (Leslie aging, fertility, natural mortality on 3.4M agents)
            demo_results = self.demographics.step(year, mods, abm=self.abm)
            self.refugee_pool = demo_results['refugees_remaining']
            
            # 3. Frontline Displacement (IDPs flee occupied regions)
            occupied_indices = [self.regions.index(r) for r in self.regions if self.frontline_states[r] == 2]
            safe_indices = [self.regions.index(r) for r in self.regions if self.frontline_states[r] == 0]
            if len(safe_indices) == 0:
                safe_indices = [r_idx for r_idx in range(self.R) if r_idx not in occupied_indices]
                
            if len(occupied_indices) > 0:
                in_occupied = np.isin(self.abm.agent_region, occupied_indices) & (self.abm.agent_health != 2)
                N_flee = np.sum(in_occupied)
                if N_flee > 0:
                    self.abm.agent_region[in_occupied] = np.random.choice(safe_indices, size=N_flee)

            # 4. Energy Grid Constraints Update
            for r_idx, r in enumerate(self.regions):
                state = self.frontline_states[r]
                for s in self.sectors:
                    rec = mods.get('energy_recovery', 0.04)
                    dmg = mods['war_damage'].get(r, {}).get(s, 0.0)
                    curr_util = self.energy_utilization[r][s]
                    next_util = np.clip(curr_util + rec - dmg, 0.20, 1.0)
                    
                    if state == 1:
                        next_util = min(0.50, next_util) # Capped at 50%
                    elif state == 2:
                        next_util = 0.0 # Capped at 0%
                    self.energy_utilization[r][s] = next_util

            # 5. Apply Human Capital modifier (HCI) to regional TFP
            regional_tfp = copy.deepcopy(self.tfp)
            for r_idx, r in enumerate(self.regions):
                state = self.frontline_states[r]
                
                # Calculate disability and skill shares
                r_pop = self.demographics.pop_array[r_idx]
                total_p = np.sum(r_pop)
                if total_p > 0:
                    disabled_p = np.sum(r_pop[:, :, 1])
                    disabled_rate = disabled_p / total_p
                else:
                    disabled_rate = 0.0
                    
                # Skilled labor rate from ABM agents in region
                r_mask = (self.abm.agent_region == r_idx) & (self.abm.agent_health != 2)
                total_agents = np.sum(r_mask)
                if total_agents > 0:
                    skilled_agents = np.sum(r_mask & (self.abm.agent_labor == 2))
                    skilled_rate = skilled_agents / total_agents
                else:
                    skilled_rate = 0.15
                    
                hci = (1.0 - 0.20 * disabled_rate) * (1.0 + 0.15 * skilled_rate)
                
                frontline_mult = 1.0
                if state == 1:
                    frontline_mult = 0.60 # TFP reduced by 40%
                elif state == 2:
                    frontline_mult = 0.00 # Occupied: no production
                    
                for s in self.sectors:
                    regional_tfp[r][s] *= (hci * frontline_mult)

            # 6. Micro-ABM Step (Household consumption and migration decisions)
            labor_supply, aggregate_consumption = self.abm.step(
                prices=self.prices,
                wages_by_type=self.wages_by_type,
                grp_per_capita=grp_per_capita,
                tax_rates={r: self.finance.tax_pit + self.finance.tax_military for r in self.regions},
                subsistence_demands=self.subsistence_demands,
                budget_shares=self.budget_shares,
                scenario_modifiers=mods
            )
            
            # 7. Macro-CGE Market Clearing Step (Solve CGE clearing equations)
            realized_output, prices_solved, wages_solved = self.cge.solve_equilibrium(
                capital=self.capital,
                labor_supply_by_type=labor_supply,
                tfp=regional_tfp,
                prices_init=self.prices,
                energy_utilization=self.energy_utilization,
                household_demands=aggregate_consumption,
                exchange_rate=self.finance.exchange_rate,
                interest_rate=self.finance.interest_rate,
                p_world_import=mods.get('p_world_import', None),
                p_world_export=mods.get('p_world_export', None)
            )
            
            self.prices = prices_solved
            self.wages_by_type = wages_solved

            # 8. Financial aggregates
            total_wages = 0.0
            total_profits = 0.0
            regional_grp_nominal = {r: 0.0 for r in self.regions}
            regional_grp_real = {r: 0.0 for r in self.regions}
            
            for r in self.regions:
                state = self.frontline_states[r]
                if state == 2:
                    continue
                    
                ws = self.wages_by_type[r]['skilled']
                wm = self.wages_by_type[r]['semi-skilled']
                wu = self.wages_by_type[r]['unskilled']
                
                ls = labor_supply[r]['skilled']
                lm = labor_supply[r]['semi-skilled']
                lu = labor_supply[r]['unskilled']
                
                lab_cost = (ls * ws) + (lm * wm) + (lu * wu)
                total_wages += lab_cost
                
                r_dest_idx = self.regions.index(r)
                for s in self.sectors:
                    out_val_nominal = realized_output[(r, s)] * self.prices[r][s]
                    
                    int_cost_nominal = 0.0
                    for sx_idx, sx in enumerate(self.sectors):
                        coeff = self.base_tech.get(s, {}).get(sx, 0.0)
                        if coeff > 0:
                            qty_needed = realized_output[(r, s)] * coeff
                            for rj in range(self.R):
                                share = self.cge.trade_shares[rj, r_dest_idx, sx_idx]
                                p_source = self.prices[self.regions[rj]][sx]
                                int_cost_nominal += share * qty_needed * p_source
                                
                    profit = max(0.0, out_val_nominal - int_cost_nominal)
                    total_profits += profit
                    regional_grp_nominal[r] += out_val_nominal - int_cost_nominal
                    
                    # Real (base-price = 1.0) GRP
                    out_val_real = realized_output[(r, s)] * 1.0
                    int_cost_real = sum(realized_output[(r, s)] * self.base_tech.get(s, {}).get(sx, 0.0) for sx in self.sectors)
                    regional_grp_real[r] += out_val_real - int_cost_real

            # Sum regional GRPs to national GDP
            real_gdp_uah = sum(regional_grp_real.values())
            nominal_gdp_uah = sum(regional_grp_nominal.values())
            nominal_gdp_usd = nominal_gdp_uah / self.finance.exchange_rate
            
            # Export / Import totals (USD) from endogenous trade
            total_exports_usd = sum(self.cge.realized_exports[(r, s)] for r in self.regions for s in self.sectors if self.frontline_states[r] != 2) / self.finance.exchange_rate
            total_imports_usd = sum(self.cge.realized_imports[(r, s)] for r in self.regions for s in self.sectors if self.frontline_states[r] != 2) / self.finance.exchange_rate
            
            # Living agents wealth
            household_wealth_sum = float(np.sum(self.abm.agent_wealth[self.abm.agent_health != 2]))

            # Calculate pensioners: working-age disabled + elderly residing in Ukraine
            if self.abm is not None:
                residing = self.abm.agent_health <= 1
                num_pensioners = float(np.sum(residing & ((self.abm.agent_cohort > 12) | (self.abm.agent_health == 1))))
            else:
                num_pensioners = float(np.sum(self.demographics.pop_array[:, 13:, :, :]) + np.sum(self.demographics.pop_array[:, :13, :, 1]))

            # Step Macro-Finance engine
            fin_results = self.finance.step(
                year=year,
                nominal_gdp_uah=nominal_gdp_uah,
                nominal_gdp_usd=nominal_gdp_usd,
                total_wages=total_wages,
                corporate_profits=total_profits,
                exports_usd=total_exports_usd,
                imports_usd=total_imports_usd,
                scenario_modifiers=mods,
                household_wealth_sum=household_wealth_sum,
                num_pensioners=num_pensioners
            )
            
            # 9. Dynamic Capital Accumulation with Bank Lending
            bank_loans_added = fin_results['bank_loans']
            fdi_uah = mods.get('fdi_usd', 1.5e9) * fin_results['exchange_rate']
            aid_loans_uah = mods.get('foreign_aid_usd', 22.0e9) * (1.0 - mods.get('foreign_aid_grant_share', 0.50)) * fin_results['exchange_rate']
            
            # Interest rates affect profit reinvestment share and bank loan investment share
            reinvested_profits_share = np.clip(0.30 - 0.8 * fin_results['interest_rate'], 0.10, 0.25)
            reinvested_profits = total_profits * reinvested_profits_share
            bank_inv_share = np.clip(0.10 - 0.3 * fin_results['interest_rate'], 0.02, 0.08)
            
            total_investment = fdi_uah + aid_loans_uah + reinvested_profits + bank_loans_added * bank_inv_share
            
            investment_allocation = {}
            for r in self.regions:
                investment_allocation[r] = {}
                state = self.frontline_states[r]
                if state == 2:
                    for s in self.sectors:
                        investment_allocation[r][s] = 0.0
                    continue
                    
                grp_share = regional_grp_real[r] / max(1e-5, real_gdp_uah)
                for s in self.sectors:
                    sector_share = realized_output[(r, s)] / max(1e-5, sum(realized_output[(r, sx)] for sx in self.sectors))
                    investment_allocation[r][s] = total_investment * grp_share * sector_share

            # Update capital stocks
            self.capital = self.production.accumulate_capital(
                capital=self.capital,
                investment=investment_allocation,
                war_damage=mods['war_damage']
            )
            
            for r in self.regions:
                state = self.frontline_states[r]
                if state == 1:
                    for s in self.sectors:
                        self.capital[r][s] = max(1e-3, self.capital[r][s] * (1.0 - 0.15))
                elif state == 2:
                    for s in self.sectors:
                        self.capital[r][s] = 1e-3
            
            # Sync capital back to Firm Agents
            for r in self.regions:
                for s in self.sectors:
                    self.abm.firms[r][s].capital = self.capital[r][s]
            
            # TFP growth step: region-specific tfp growth based on frontline status
            for r in self.regions:
                state = self.frontline_states[r]
                if state == 0:
                    growth_rate = mods.get('tfp_growth', 0.018)
                elif state == 1:
                    growth_rate = -0.05
                else: # state == 2
                    growth_rate = 0.0
                    
                for s in self.sectors:
                    self.tfp[r][s] *= (1.0 + growth_rate)

            # Store year snapshot
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
                'regional_data': {r: {
                    'grp_real': regional_grp_real[r],
                    'grp_nominal': regional_grp_nominal[r],
                    'pop': sum(self.demographics.pop[r][g].sum() for g in ['Male', 'Female']),
                    'wage_skilled': self.wages_by_type[r]['skilled'],
                    'wage_unskilled': self.wages_by_type[r]['unskilled'],
                    'frontline_state': int(self.frontline_states[r]),
                    'sectors': {s: {
                        'output': realized_output[(r, s)],
                        'capital': self.capital[r][s],
                        'price': self.prices[r][s]
                    } for s in self.sectors}
                } for r in self.regions}
            }
            history.append(snapshot)
            
        return history
