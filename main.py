import argparse
import json
import os
import sys
import numpy as np

from data.loader import generate_baseline_data
from engine.model import ModelRunner
from engine.scenarios import ScenarioEngine
from benchmark import run_performance_profile

def convert_to_json_serializable(obj):
    """
    Recursively converts dictionary keys to string and NumPy types to native python types.
    """
    if isinstance(obj, dict):
        # Convert tuple keys (like (region, sector) in target demand) to string representation
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                key_str = f"{k[0]}|{k[1]}"
            else:
                key_str = str(k)
            new_dict[key_str] = convert_to_json_serializable(v)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

def run_single_scenario(scenario_name, base_data, num_years=25):
    runner = ModelRunner(base_data)
    scenarios = ScenarioEngine(runner.regions, runner.sectors)
    print(f"\nRunning deterministic simulation for scenario: '{scenario_name.upper()}' (2026-2050)")
    print("-" * 80)
    history = runner.run_simulation(scenario_name, scenarios, num_years=num_years)
    
    # Print yearly console log summary
    print(f"{'Year':<6} | {'GDP Real (Trln UAH)':<20} | {'GDP Nom (Bln USD)':<18} | {'Pop (Mln)':<10} | {'Skilled Wage':<13} | {'Unskilled Wage':<15} | {'Debt/GDP':<10}")
    print("-" * 110)
    for snap in history:
        y = snap['year']
        gdp_uah = snap['gdp_real_uah'] / 1.0e12
        gdp_usd = snap['gdp_nominal_usd'] / 1.0e9
        pop = snap['population'] / 1.0e6
        debt = snap['debt_gdp'] * 100.0
        
        # Calculate average wages across regions
        avg_skilled = sum(r_data['wage_skilled'] for r_data in snap['regional_data'].values()) / len(snap['regional_data'])
        avg_unskilled = sum(r_data['wage_unskilled'] for r_data in snap['regional_data'].values()) / len(snap['regional_data'])
        
        print(f"{y:<6} | {gdp_uah:<20.3f} | {gdp_usd:<18.2f} | {pop:<10.2f} | {avg_skilled:<13.1f} | {avg_unskilled:<15.1f} | {debt:<9.1f}%")
    print("-" * 110)
    return history

def run_monte_carlo(scenario_name, base_data, num_trials, num_years=25):
    print(f"\nRunning Monte Carlo LHS simulation for scenario: '{scenario_name.upper()}' ({num_trials} trials)")
    print("-" * 80)
    
    # Instantiate models
    runner_ref = ModelRunner(base_data)
    scenarios = ScenarioEngine(runner_ref.regions, runner_ref.sectors)
    
    # Generate LHS samples
    lhs_samples = scenarios.generate_lhs_samples(num_trials)
    
    # Run trials and store trajectories
    gdp_trajectories = np.zeros((num_trials, num_years))
    pop_trajectories = np.zeros((num_trials, num_years))
    debt_trajectories = np.zeros((num_trials, num_years))
    inflation_trajectories = np.zeros((num_trials, num_years))
    
    all_trial_data = []
    
    t_start = os.times().elapsed
    for trial in range(num_trials):
        # Fresh initialization for each trial
        runner = ModelRunner(base_data)
        sample = lhs_samples[trial]
        history = runner.run_simulation(scenario_name, scenarios, num_years=num_years, lhs_sample=sample)
        
        # Extract variables
        for yr_idx, snap in enumerate(history):
            gdp_trajectories[trial, yr_idx] = snap['gdp_real_uah']
            pop_trajectories[trial, yr_idx] = snap['population']
            debt_trajectories[trial, yr_idx] = snap['debt_gdp']
            inflation_trajectories[trial, yr_idx] = snap['inflation']
            
        all_trial_data.append({
            'trial_idx': trial,
            'sample_parameters': sample,
            'gdp': gdp_trajectories[trial].tolist()
        })
        
        if (trial + 1) % max(1, num_trials // 5) == 0 or trial == num_trials - 1:
            print(f"-> Completed {trial + 1}/{num_trials} trials...")
            
    # Calculate percentiles (10th, 50th, 90th)
    percentiles = [10, 50, 90]
    gdp_p = {p: np.percentile(gdp_trajectories, p, axis=0) for p in percentiles}
    pop_p = {p: np.percentile(pop_trajectories, p, axis=0) for p in percentiles}
    debt_p = {p: np.percentile(debt_trajectories, p, axis=0) for p in percentiles}
    inf_p = {p: np.percentile(inflation_trajectories, p, axis=0) for p in percentiles}
    
    # Print summary highlights
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION SUMMARY RESULTS (Real GDP in Trillion UAH)")
    print("=" * 80)
    print(f"{'Year':<6} | {'10th Percentile (P10)':<22} | {'50th Percentile (P50)':<22} | {'90th Percentile (P90)':<22}")
    print("-" * 80)
    
    highlight_years = [2026, 2030, 2035, 2040, 2045, 2050]
    for yr in highlight_years:
        idx = yr - 2026
        print(f"{yr:<6} | {gdp_p[10][idx]/1e12:<22.3f} | {gdp_p[50][idx]/1e12:<22.3f} | {gdp_p[90][idx]/1e12:<22.3f}")
    print("=" * 80)
    
    return {
        'scenario': scenario_name,
        'num_trials': num_trials,
        'gdp_percentiles': gdp_p,
        'population_percentiles': pop_p,
        'debt_percentiles': debt_p,
        'inflation_percentiles': inf_p,
        'trials': all_trial_data
    }

def main():
    parser = argparse.ArgumentParser(description="UkrEcoSim2050: Ukraine Detailed Economic Simulation CLI")
    parser.add_argument('--scenario', type=str, choices=['baseline', 'optimistic', 'pessimistic', 'all'], default='baseline',
                        help="Choose simulation scenario ('all' runs all three deterministically)")
    parser.add_argument('--monte-carlo', action='store_true', help="Run Monte Carlo simulation with stochastic shocks")
    parser.add_argument('--trials', type=int, default=50, help="Number of Monte Carlo trials")
    parser.add_argument('--out', type=str, default='simulation_results.json', help="File path to save JSON results")
    parser.add_argument('--benchmark', action='store_true', help="Run performance benchmarks and scaling reports")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_performance_profile()
        sys.exit(0)
        
    # Generate common baseline 2026 data
    base_data = generate_baseline_data()
    
    results = {}
    
    if args.monte_carlo:
        # Run Monte Carlo for the chosen scenario (or baseline by default)
        scen = 'baseline' if args.scenario == 'all' else args.scenario
        results['monte_carlo'] = run_monte_carlo(scen, base_data, args.trials)
    else:
        # Run deterministic simulation(s)
        if args.scenario == 'all':
            for s in ['baseline', 'optimistic', 'pessimistic']:
                results[s] = run_single_scenario(s, base_data)
        else:
            results[args.scenario] = run_single_scenario(args.scenario, base_data)
            
    # Save results to JSON
    print(f"\nSerializing and saving results to: {args.out}...")
    serializable_results = convert_to_json_serializable(results)
    
    try:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print("-> Results saved successfully!")
    except Exception as e:
        print(f"-> Error saving results: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
