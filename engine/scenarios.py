import numpy as np

class ScenarioEngine:
    """
    Manages deterministic scenarios (Optimistic, Baseline, Pessimistic)
    and generates stochastic shocks using Latin Hypercube Sampling (LHS).
    Supports 6 uncertainty dimensions (including world commodity prices and interest rate shocks).
    """
    def __init__(self, regions, sectors):
        self.regions = regions
        self.sectors = sectors

    def get_deterministic_modifiers(self, scenario_name, year):
        """
        Returns the base parameters for the given scenario and year.
        Projected years: 2026 to 2050.
        """
        # Time-based interpolation factor (0.0 at 2026, 1.0 at 2050)
        t = (year - 2026) / 24.0
        
        modifiers = {}
        
        if scenario_name == 'optimistic':
            # Rapid recovery, EU integration
            modifiers['repatriation_rate'] = 0.08 - 0.04 * t # Returns start high, taper off as refugees deplete
            modifiers['mobilization_rate'] = max(0.005, 0.04 * (1.0 - t * 2.0)) if year < 2038 else 0.005
            modifiers['labor_participation'] = 0.72 + 0.03 * t
            modifiers['brain_drain_rate'] = max(0.001, 0.008 * (1.0 - t * 3.0))
            modifiers['fertility_modifier'] = 1.0 + 0.25 * t # Post-war baby boom
            modifiers['remittance_rate_per_worker'] = 0.05  # Low remittances (most return home)
            
            # TFP growth (high)
            modifiers['tfp_growth'] = 0.035
            
            # Fiscal & reconstruction
            modifiers['defense_spending_ratio'] = max(0.04, 0.25 - 0.21 * (year - 2026)/10.0) if year < 2036 else 0.04
            modifiers['social_spending_ratio'] = 0.16 + 0.02 * t
            modifiers['reconstruction_needs_usd'] = 20.0e9 * (1.0 - t) if year < 2046 else 1.0e9 # Reconstruction wraps up
            modifiers['foreign_aid_usd'] = max(5.0e9, 35.0e9 * (1.0 - t * 1.5))
            modifiers['foreign_aid_grant_share'] = 0.75
            modifiers['fdi_usd'] = 3.0e9 + 12.0e9 * t # FDI surges as security is guaranteed
            modifiers['deficit_monetization_rate'] = max(0.0, 0.05 * (1.0 - t * 2.0))
            
            # Energy grid recovery
            modifiers['energy_recovery'] = 0.08 # 8% recovery of damaged grid capacity per year
            
        elif scenario_name == 'pessimistic':
            # Frozen conflict, prolonged war of attrition
            modifiers['repatriation_rate'] = 0.01
            modifiers['mobilization_rate'] = 0.05 # Stays high
            modifiers['labor_participation'] = 0.65
            modifiers['brain_drain_rate'] = 0.01 # Steady loss of young talent
            modifiers['fertility_modifier'] = 0.85
            modifiers['remittance_rate_per_worker'] = 0.15  # High remittances (emigrants stay)
            
            # TFP growth (stagnant)
            modifiers['tfp_growth'] = 0.005
            
            # Fiscal & reconstruction
            modifiers['defense_spending_ratio'] = 0.28 # Remains extremely high
            modifiers['social_spending_ratio'] = 0.12
            modifiers['reconstruction_needs_usd'] = 10.0e9 # Constantly repairing damage
            modifiers['foreign_aid_usd'] = 12.0e9 # Just enough to survive
            modifiers['foreign_aid_grant_share'] = 0.30 # Mostly loans, leading to debt trap
            modifiers['fdi_usd'] = 0.5e9
            modifiers['deficit_monetization_rate'] = 0.20 # Monetizing deficits to fund defense
            
            # Energy grid recovery
            modifiers['energy_recovery'] = 0.01
            
        else: # baseline
            # Gradual recovery, slow reforms
            modifiers['repatriation_rate'] = 0.04 - 0.02 * t
            modifiers['mobilization_rate'] = max(0.015, 0.04 * (1.0 - t * 1.5)) if year < 2040 else 0.015
            modifiers['labor_participation'] = 0.69 + 0.02 * t
            modifiers['brain_drain_rate'] = max(0.003, 0.006 * (1.0 - t * 2.0))
            modifiers['fertility_modifier'] = 1.0 + 0.10 * t
            modifiers['remittance_rate_per_worker'] = 0.10  # Moderate remittances
            
            # TFP growth (moderate)
            modifiers['tfp_growth'] = 0.018
            
            # Fiscal & reconstruction
            modifiers['defense_spending_ratio'] = max(0.08, 0.25 - 0.17 * (year - 2026)/12.0) if year < 2038 else 0.08
            modifiers['social_spending_ratio'] = 0.14 + 0.02 * t
            modifiers['reconstruction_needs_usd'] = 15.0e9 * (1.0 - t * 0.8)
            modifiers['foreign_aid_usd'] = max(3.0e9, 22.0e9 * (1.0 - t * 1.2))
            modifiers['foreign_aid_grant_share'] = 0.50
            modifiers['fdi_usd'] = 1.5e9 + 4.5e9 * t
            modifiers['deficit_monetization_rate'] = max(0.01, 0.08 * (1.0 - t * 1.8))
            
            # Energy grid recovery
            modifiers['energy_recovery'] = 0.04
            
        # Populate region-specific TFP growth rates
        tfp_growth_by_region = {}
        for r in self.regions:
            if r in ['Donetsk', 'Luhansk', 'Crimea', 'Sevastopol']:
                tfp_growth_by_region[r] = 0.0 # Occupied
            elif r in ['Kharkiv', 'Zaporizhzhia', 'Kherson', 'Mykolaiv']:
                tfp_growth_by_region[r] = -0.05 # Frontline
            else:
                tfp_growth_by_region[r] = modifiers['tfp_growth']
        modifiers['tfp_growth_by_region'] = tfp_growth_by_region
            
        return modifiers

    def generate_lhs_samples(self, num_trials):
        """
        Generates Latin Hypercube Samples for Monte Carlo runs.
        We have 6 key uncertain variables:
        1. TFP shock factor (Normal distribution around scenario mean)
        2. War damage intensity (Log-normal distribution)
        3. Global export demand (Normal distribution)
        4. Foreign aid realization (Uniform distribution)
        5. NBU interest rate policy shock (Uniform shift)
        6. Global commodity world price shock (Normal distribution)
        
        Returns:
            - Dict of trial_idx -> dict of sampled values
        """
        num_vars = 6
        # Create empty LHS design matrix
        design = np.zeros((num_trials, num_vars))
        
        for v in range(num_vars):
            # Divide space [0, 1] into num_trials intervals
            intervals = np.linspace(0.0, 1.0, num_trials + 1)
            # Sample uniformly within each interval
            lows = intervals[:-1]
            highs = intervals[1:]
            points = lows + np.random.rand(num_trials) * (highs - lows)
            # Shuffle the points to randomize indices across variables
            np.random.shuffle(points)
            design[:, v] = points
            
        # Map design matrix columns to actual distributions
        samples = {}
        for trial in range(num_trials):
            # 1. TFP shock (centered at 1.0, std dev 0.02)
            tfp_p = design[trial, 0]
            tfp_shock = float(np.percentile(np.random.normal(1.0, 0.02, 10000), tfp_p * 100))
            
            # 2. War damage
            wd_p = design[trial, 1]
            wd_shock = float(np.percentile(np.random.lognormal(-5.0, 0.8, 10000), wd_p * 100))
            
            # 3. Export demand
            exp_p = design[trial, 2]
            export_shock = float(np.percentile(np.random.normal(1.0, 0.08, 10000), exp_p * 100))
            
            # 4. Foreign aid multiplier (0.8x to 1.2x of projected aid)
            aid_p = design[trial, 3]
            aid_multiplier = 0.8 + 0.4 * aid_p
            
            # 5. NBU interest rate policy shock (-3.0% to +5.0% shift)
            ir_p = design[trial, 4]
            interest_rate_shock = -0.03 + 0.08 * ir_p
            
            # 6. Global commodity world price shock (0.7x to 1.4x of base)
            wp_p = design[trial, 5]
            world_price_shock = 0.7 + 0.7 * wp_p
            
            samples[trial] = {
                'tfp_shock': tfp_shock,
                'war_damage_intensity': wd_shock,
                'export_demand_shock': export_shock,
                'aid_multiplier': aid_multiplier,
                'interest_rate_shock': interest_rate_shock,
                'world_price_shock': world_price_shock
            }
            
        return samples

    def apply_stochastic_shocks(self, base_modifiers, lhs_sample):
        """
        Combines deterministic modifiers with Monte Carlo LHS samples.
        """
        mods = base_modifiers.copy()
        
        # Apply TFP modifier
        mods['tfp_growth'] += (lhs_sample['tfp_shock'] - 1.0) * 0.1
        
        # Apply stochastic TFP growth by region
        tfp_growth_by_region = {}
        for r in self.regions:
            state = mods.get('frontline_states', {}).get(r, 0)
            if state == 0:
                tfp_growth_by_region[r] = mods['tfp_growth']
            elif state == 1:
                tfp_growth_by_region[r] = -0.05 + (lhs_sample['tfp_shock'] - 1.0) * 0.05
            else: # state == 2
                tfp_growth_by_region[r] = 0.0
        mods['tfp_growth_by_region'] = tfp_growth_by_region
        
        # Apply foreign aid multiplier
        mods['foreign_aid_usd'] *= lhs_sample['aid_multiplier']
        
        # Apply interest rate shock
        mods['interest_rate_shock'] = lhs_sample['interest_rate_shock']
        
        # Construct world import and export prices vectors based on global price shock
        p_world_export = np.ones(len(self.sectors))
        p_world_import = np.ones(len(self.sectors))
        
        wp_shock = lhs_sample['world_price_shock']
        commodity_sectors = [
            'AgriGrain', 'AgriTechnical', 'SteelIron', 'MetalProducts', 'CoalMining', 
            'OilGasExtraction', 'Agriculture', 'Metallurgy', 'Energy'
        ]
        for s_idx, s in enumerate(self.sectors):
            if s in commodity_sectors:
                p_world_export[s_idx] *= wp_shock
                p_world_import[s_idx] *= wp_shock
                
        mods['p_world_export'] = p_world_export
        mods['p_world_import'] = p_world_import
        
        # Add war damage details to regions/sectors
        war_damage = {}
        intensity = lhs_sample['war_damage_intensity']
        
        # Eastern/Southern regions: Donetsk, Luhansk, Kharkiv, Zaporizhzhia, Kherson, Mykolaiv, Odesa
        high_risk_regions = ['Donetsk', 'Luhansk', 'Kharkiv', 'Zaporizhzhia', 'Kherson', 'Mykolaiv', 'Odesa']
        
        for r in self.regions:
            war_damage[r] = {}
            multiplier = 3.0 if r in high_risk_regions else 0.2
            for s in self.sectors:
                # Energy and Metallurgy are highly targeted
                sector_mult = 2.0 if s in ['Energy', 'Metallurgy', 'Transport', 'SteelIron', 'EnergyThermal', 'EnergyNuclearGen'] else 0.5
                war_damage[r][s] = intensity * multiplier * sector_mult
                
        mods['war_damage'] = war_damage
        mods['export_shock'] = lhs_sample['export_demand_shock']
        
        return mods
