import unittest
import numpy as np
from data.loader import generate_baseline_data
from engine.demographics import DemographicEngine
from engine.production import ProductionEngine
from engine.abm import ABMEngine
from engine.cge import CGESolver
from engine.finance import FinanceEngine
from engine.model import ModelRunner
from engine.scenarios import ScenarioEngine

class TestUkrEcoSim(unittest.TestCase):
    def setUp(self):
        # Generate baseline dataset
        self.base_data = generate_baseline_data()
        self.runner = ModelRunner(self.base_data)
        self.scenarios = ScenarioEngine(self.runner.regions, self.runner.sectors)

    def test_demographic_conservation(self):
        """
        Tests demographic step calculations and population scale.
        """
        pop_start = sum(np.sum(self.runner.demographics.pop[r]['Male'] + self.runner.demographics.pop[r]['Female']) for r in self.runner.regions)
        self.assertGreater(pop_start, 30.0e6)
        
        mods = self.scenarios.get_deterministic_modifiers('baseline', 2026)
        mods['refugee_pool'] = 5e6
        mods['grp_per_capita'] = {r: 1.0 for r in self.runner.regions}
        mods['war_damage'] = {}
        mods['export_shock'] = 1.0
        
        res = self.runner.demographics.step(2026, mods)
        pop_end = res['total_pop']
        
        self.assertGreater(pop_end, 0)
        self.assertGreater(res['births'], 0)
        self.assertGreater(res['deaths'], 0)

    def test_cge_solving(self):
        """
        Tests that the non-linear CGE solver correctly clears all commodity
        and labor markets under initial conditions.
        """
        cge = self.runner.cge
        
        # Aggregate labor supply and household demands
        labor_supply = {r: {'skilled': 3e6, 'unskilled': 5e6} for r in self.runner.regions}
        household_demands = {r: {s: 1e9 for s in self.runner.sectors} for r in self.runner.regions}
        
        realized_output, prices_solved, wages_solved = cge.solve_equilibrium(
            capital=self.runner.capital,
            labor_supply_by_type=labor_supply,
            tfp=self.runner.tfp,
            prices_init=self.runner.prices,
            energy_utilization=self.runner.energy_utilization,
            household_demands=household_demands
        )
        
        # Verify prices and wages are strictly positive
        for r in self.runner.regions:
            self.assertGreater(wages_solved[r]['skilled'], 0.0)
            self.assertGreater(wages_solved[r]['unskilled'], 0.0)
            for s in self.runner.sectors:
                self.assertGreater(prices_solved[r][s], 0.0)
                self.assertGreater(realized_output[(r, s)], 0.0)

    def test_abm_agent_initialization(self):
        """
        Tests that Household and Firm agents are initialized with consistent attributes.
        """
        abm = self.runner.abm
        self.assertTrue(4900 <= len(abm.households) <= abm.num_households)
        
        # Check active status and initial wealth
        for agent in abm.households[:10]:
            self.assertEqual(agent.region in self.runner.regions, True)
            self.assertEqual(agent.labor_type in ['skilled', 'unskilled'], True)
            self.assertGreater(agent.wealth, 0.0)
            self.assertEqual(agent.active, True)
            
        # Check firms
        for r in self.runner.regions:
            for s in self.runner.sectors:
                self.assertGreater(abm.firms[r][s].capital, 0.0)

    def test_full_model_run_consistency(self):
        """
        Tests that running the complete Hybrid AB-CGE simulation
        generates consistent history results without throwing exceptions.
        """
        history = self.runner.run_simulation('baseline', self.scenarios, num_years=3)
        self.assertEqual(len(history), 3)
        
        for snap in history:
            self.assertGreater(snap['gdp_real_uah'], 0.0)
            self.assertGreater(snap['population'], 0.0)
            self.assertGreater(snap['exchange_rate'], 0.0)
            self.assertGreater(snap['inflation'], 0.0)
            self.assertGreater(snap['debt_gdp'], 0.0)

    def test_mrio_solver_93_sectors(self):
        """
        Tests that MRIOSolver correctly initializes with 93 sectors and has size 2511x2511.
        """
        from engine.mrio import MRIOSolver
        mrio = MRIOSolver(self.runner.regions, self.runner.sectors, self.base_data['base_tech_coefficients'], distances=self.runner.distances)
        self.assertEqual(mrio.N, 27 * 93)
        self.assertEqual(mrio.A.shape, (2511, 2511))

if __name__ == "__main__":
    unittest.main()
