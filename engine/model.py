import numpy as np
import copy
from engine.demographics import DemographicEngine
from engine.production import ProductionEngine
from engine.abm import ABMEngine
from engine.cge import CGESolver
from engine.finance import FinanceEngine

class ModelRunner:
    """
    Main controller for the Hybrid AB-CGE simulator.
    Coordinates Demographics, Micro-ABM, Macro-CGE, and Finance engines.
    """
    def __init__(self, base_data):
        self.regions = base_data['regions']
        self.sectors = base_data['sectors']
        self.initial_pop = copy.deepcopy(base_data['initial_pop'])
        
        # Initialize Demographics
        self.demographics = DemographicEngine(
            regions=self.regions,
            initial_pop=self.initial_pop,
            fertility_rates=copy.deepcopy(base_data['fertility_rates']),
            mortality_rates=copy.deepcopy(base_data['mortality_rates']),
            migration_gravity_coeffs=copy.deepcopy(base_data['migration_gravity'])
        )
        
        # Initialize Production
        self.production = ProductionEngine(
            regions=self.regions,
            sectors=self.sectors
        )
        
        # Initialize ABM Engine & Agents
        self.abm = ABMEngine(self.regions, self.sectors)
        self.capital = copy.deepcopy(base_data['initial_capital'])
        self.abm.initialize_agents(self.initial_pop, self.capital)
        
        # Initialize CGE Solver
        self.base_tech = copy.deepcopy(base_data['base_tech_coefficients'])
        self.cge = CGESolver(self.regions, self.sectors, self.base_tech)
        
        self.finance = FinanceEngine()
        
        # Core State Variables
        self.tfp = copy.deepcopy(base_data['initial_tfp'])
        self.prices = copy.deepcopy(base_data['prices'])
        self.energy_utilization = copy.deepcopy(base_data['energy_utilization'])
        
        self.budget_shares = copy.deepcopy(base_data['budget_shares'])
        self.subsistence_demands = copy.deepcopy(base_data['subsistence_demands'])
        self.wages_by_type = copy.deepcopy(base_data['wages_by_type'])
        
        self.refugee_pool = 5.0e6

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
                
            # 2. Demographic Step
            demo_results = self.demographics.step(year, mods)
            self.refugee_pool = demo_results['total_pop'] * 0.15 # Refugee tracking
            
            # 3. Energy Grid Constraints Update
            for r in self.regions:
                for s in self.sectors:
                    rec = mods.get('energy_recovery', 0.04)
                    dmg = mods['war_damage'].get(r, {}).get(s, 0.0)
                    curr_util = self.energy_utilization[r][s]
                    self.energy_utilization[r][s] = np.clip(curr_util + rec - dmg, 0.20, 1.0)

            # 4. Micro-ABM Step
            # Step household agents: logit migration, Stone-Geary consumption, labor supply splits
            labor_supply, aggregate_consumption = self.abm.step(
                prices=self.prices,
                wages_by_type=self.wages_by_type,
                grp_per_capita=grp_per_capita,
                tax_rates={r: self.finance.tax_pit + self.finance.tax_military for r in self.regions},
                subsistence_demands=self.subsistence_demands,
                budget_shares=self.budget_shares,
                scenario_modifiers=mods
            )
            
            # 5. Macro-CGE Market Clearing Step
            # Solve for equilibrium prices and wages clearing all 459 commodity and labor markets
            realized_output, prices_solved, wages_solved = self.cge.solve_equilibrium(
                capital=self.capital,
                labor_supply_by_type=labor_supply,
                tfp=self.tfp,
                prices_init=self.prices,
                energy_utilization=self.energy_utilization,
                household_demands=aggregate_consumption
            )
            
            # Update state with solved equilibrium prices and wages
            self.prices = prices_solved
            self.wages_by_type = wages_solved

            # 6. Aggregate Financial flows
            total_wages = 0.0
            total_profits = 0.0
            regional_grp_nominal = {r: 0.0 for r in self.regions}
            regional_grp_real = {r: 0.0 for r in self.regions}
            
            # Map region index helper
            node_to_idx = self.cge.node_to_idx
            
            for r in self.regions:
                ws = self.wages_by_type[r]['skilled']
                wu = self.wages_by_type[r]['unskilled']
                
                # Fetch skilled/unskilled labor counts from ABM
                ls = labor_supply[r]['skilled']
                lu = labor_supply[r]['unskilled']
                
                lab_cost = (ls * ws) + (lu * wu)
                total_wages += lab_cost
                
                # Intermediate demands from CGE
                r_dest_idx = self.regions.index(r)
                for s in self.sectors:
                    out_val_nominal = realized_output[(r, s)] * self.prices[r][s]
                    
                    # Estimate intermediate input cost using solved prices
                    int_cost_nominal = 0.0
                    for sx_idx, sx in enumerate(self.sectors):
                        coeff = self.base_tech.get(s, {}).get(sx, 0.0)
                        if coeff > 0:
                            qty_needed = realized_output[(r, s)] * coeff
                            for rj in range(len(self.regions)):
                                share = self.cge.trade_shares[rj, r_dest_idx, sx_idx]
                                p_source = self.prices[self.regions[rj]][sx]
                                int_cost_nominal += share * qty_needed * p_source
                    
                    profit = max(0.0, out_val_nominal - int_cost_nominal)
                    total_profits += profit
                    regional_grp_nominal[r] += out_val_nominal - int_cost_nominal
                    
                    # Real output and intermediate costs (using base prices = 1.0)
                    out_val_real = realized_output[(r, s)] * 1.0
                    int_cost_real = sum(realized_output[(r, s)] * self.base_tech.get(s, {}).get(sx, 0.0) for sx in self.sectors)
                    regional_grp_real[r] += out_val_real - int_cost_real

            # Sum regional GRPs to get National Real & Nominal GDP (UAH)
            real_gdp_uah = sum(regional_grp_real.values())
            nominal_gdp_uah = sum(regional_grp_nominal.values())
            nominal_gdp_usd = nominal_gdp_uah / self.finance.exchange_rate
            
            # Export / Import totals (USD)
            total_exports_usd = sum(realized_output[(r, s)] * 0.15 for r in self.regions for s in ['Agriculture', 'Metallurgy', 'IT']) / self.finance.exchange_rate
            total_imports_usd = sum(realized_output[(r, s)] * 0.18 for r in self.regions for s in self.sectors) / self.finance.exchange_rate

            # Step Macro-Finance engine
            fin_results = self.finance.step(
                year=year,
                nominal_gdp_uah=nominal_gdp_uah,
                nominal_gdp_usd=nominal_gdp_usd,
                total_wages=total_wages,
                corporate_profits=total_profits,
                exports_usd=total_exports_usd,
                imports_usd=total_imports_usd,
                scenario_modifiers=mods
            )
            
            # 7. Dynamic Capital Accumulation
            fdi_uah = mods.get('fdi_usd', 2.0e9) * fin_results['exchange_rate']
            aid_loans_uah = mods.get('foreign_aid_usd', 20.0e9) * (1.0 - mods.get('foreign_aid_grant_share', 0.50)) * fin_results['exchange_rate']
            reinvested_profits = total_profits * 0.20
            total_investment = fdi_uah + aid_loans_uah + reinvested_profits
            
            # Share investment across regional sectors
            investment_allocation = {}
            for r in self.regions:
                investment_allocation[r] = {}
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
            
            # Sync capital back to Firm Agents in the ABM
            for r in self.regions:
                for s in self.sectors:
                    self.abm.firms[r][s].capital = self.capital[r][s]
            
            # 8. TFP growth step (TFP accumulates over time based on scenario growth rates)
            for r in self.regions:
                for s in self.sectors:
                    self.tfp[r][s] *= (1.0 + mods.get('tfp_growth', 0.015))

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
                'regional_data': {r: {
                    'grp_real': regional_grp_real[r],
                    'grp_nominal': regional_grp_nominal[r],
                    'pop': sum(np.sum(self.demographics.pop[r][g]) for g in ['Male', 'Female']),
                    'wage_skilled': self.wages_by_type[r]['skilled'],
                    'wage_unskilled': self.wages_by_type[r]['unskilled'],
                    'sectors': {s: {
                        'output': realized_output[(r, s)],
                        'capital': self.capital[r][s],
                        'price': self.prices[r][s]
                    } for s in self.sectors}
                } for r in self.regions}
            }
            history.append(snapshot)
            
        return history
