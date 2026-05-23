import time
import numpy as np
from data.loader import generate_baseline_data
from engine.model import ModelRunner
from engine.scenarios import ScenarioEngine

def run_performance_profile():
    print("=" * 60)
    print("RUNNING HYBRID AB-CGE PERFORMANCE PROFILING")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Load Data
    print("Step 1: Initializing baseline data (27 regions, 15 sectors = 405 nodes)...")
    t0 = time.time()
    base_data = generate_baseline_data()
    t_data = time.time() - t0
    print(f"-> Completed in {t_data:.4f} seconds.")
    
    # 2. Setup engines
    print("Step 2: Building simulation model runner...")
    t0 = time.time()
    runner = ModelRunner(base_data)
    scenarios = ScenarioEngine(runner.regions, runner.sectors)
    t_setup = time.time() - t0
    print(f"-> Completed in {t_setup:.4f} seconds.")
    
    # 3. Profile a single year step
    print("Step 3: Profiling individual submodel step performance...")
    
    # Demographics
    t0 = time.time()
    mods = scenarios.get_deterministic_modifiers('baseline', 2026)
    mods['refugee_pool'] = 5e6
    mods['grp_per_capita'] = {r: 1.0 for r in runner.regions}
    mods['war_damage'] = {}
    mods['export_shock'] = 1.0
    demo_res = runner.demographics.step(2026, mods)
    t_demo = time.time() - t0
    
    # ABM Step
    t0 = time.time()
    labor_supply, aggregate_consumption = runner.abm.step(
        prices=runner.prices,
        wages_by_type=runner.wages_by_type,
        grp_per_capita=mods['grp_per_capita'],
        tax_rates={r: 0.23 for r in runner.regions},
        subsistence_demands=runner.subsistence_demands,
        budget_shares=runner.budget_shares,
        scenario_modifiers=mods
    )
    t_abm = time.time() - t0
    
    # CGE Equilibrium Solver
    t0 = time.time()
    realized_output, prices_solved, wages_solved = runner.cge.solve_equilibrium(
        capital=runner.capital,
        labor_supply_by_type=labor_supply,
        tfp=runner.tfp,
        prices_init=runner.prices,
        energy_utilization=runner.energy_utilization,
        household_demands=aggregate_consumption
    )
    t_cge = time.time() - t0
    
    # Finance step
    t0 = time.time()
    total_wages = 1e11
    total_profits = 5e10
    nominal_gdp_uah = 5e12
    nominal_gdp_usd = 1.2e11
    exports_usd = 2.5e10
    imports_usd = 3.0e10
    fin_res = runner.finance.step(
        year=2026,
        nominal_gdp_uah=nominal_gdp_uah,
        nominal_gdp_usd=nominal_gdp_usd,
        total_wages=total_wages,
        corporate_profits=total_profits,
        exports_usd=exports_usd,
        imports_usd=imports_usd,
        scenario_modifiers=mods
    )
    t_fin = time.time() - t0
    
    print(f"   * Demographics step time:   {t_demo*1000:.3f} ms")
    print(f"   * ABM Agent behaviors:      {t_abm*1000:.3f} ms")
    print(f"   * CGE Market-price solver:  {t_cge*1000:.3f} ms")
    print(f"   * Finance/Macro update:     {t_fin*1000:.3f} ms")
    
    # 4. Full 25-Year Run
    print("Step 4: Running full 25-year projection (2026-2050)...")
    t0 = time.time()
    # Re-initialize to clean state
    runner = ModelRunner(base_data)
    history = runner.run_simulation('baseline', scenarios, num_years=25)
    t_sim = time.time() - t0
    print(f"-> 25-year simulation completed in {t_sim:.4f} seconds.")
    print(f"   Average speed: {t_sim/25.0:.4f} seconds per simulated year.")
    
    # 5. Scaling Benchmark
    print("Step 5: Testing CGE scalability by solving synthetic matrices...")
    scaling_sectors = [15, 30, 60, 100]
    
    for s_count in scaling_sectors:
        nodes = len(runner.regions) * s_count
        # Setup dummy Jacobian size to test solving speed of non-linear systems
        A_dummy = np.random.rand(nodes, nodes) * 0.05
        for i in range(nodes):
            row_sum = np.sum(A_dummy[i, :])
            if row_sum >= 0.95:
                A_dummy[i, :] *= (0.90 / row_sum)
                
        Y_dummy = np.random.rand(nodes) * 1e6
        I_dummy = np.eye(nodes)
        
        t0 = time.time()
        X_dummy = np.linalg.solve(I_dummy - A_dummy, Y_dummy)
        t_solve = time.time() - t0
        print(f"   * CGE matrix approximation ({nodes} equations): Solved in {t_solve:.4f} seconds.")
        
    total_duration = time.time() - start_time
    print("=" * 60)
    print(f"PROFILING COMPLETE. Total benchmarking time: {total_duration:.4f} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    run_performance_profile()
