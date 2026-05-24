import numpy as np

class DemographicEngine:
    """
    Cohort-Component demographic model for 27 Ukrainian regions.
    Tracks population across 18 five-year age groups (0-4 to 85+) and 2 genders (Male, Female)
    and 2 health states (Active, Disabled).
    Integrates natural growth, Leslie matrix components, and health transitions.
    """
    def __init__(self, regions, initial_pop, fertility_rates, mortality_rates, migration_gravity_coeffs, distances=None):
        """
        initial_pop: dict of region -> {'Male': [18 cohorts], 'Female': [18 cohorts]}
        fertility_rates: dict of region -> fertility multiplier
        mortality_rates: dict of region -> {'Male': [18 values], 'Female': [18 values]}
        migration_gravity_coeffs: dict
        distances: coordinates-based distance matrix (optional)
        """
        self.regions = regions
        self.R = len(regions)
        self.gravity = migration_gravity_coeffs
        
        # 18 five-year cohorts: 0-4, 5-9, ..., 80-84, 85+
        # Health states: 0 = Active, 1 = Disabled
        # Initialize self.pop_array as a numpy array of shape (27, 18, 2, 2)
        # regions x cohorts x genders x health
        self.pop_array = np.zeros((self.R, 18, 2, 2))
        
        for r_idx, r in enumerate(regions):
            for g_idx, g in enumerate(['Male', 'Female']):
                # Distribute initial population mostly to Active health state (e.g., 96% active, 4% disabled)
                pop_data = initial_pop[r][g]
                # If the initial data has only 3 cohorts, project it to 18 cohorts
                if len(pop_data) == 3:
                    pop_18 = self._interpolate_3_to_18(pop_data)
                else:
                    pop_18 = np.array(pop_data, dtype=float)
                
                self.pop_array[r_idx, :, g_idx, 0] = pop_18 * 0.96
                self.pop_array[r_idx, :, g_idx, 1] = pop_18 * 0.04

        # Baseline age-specific fertility rates (ASFR) for cohorts 3 to 9 (ages 15-49)
        # Units: annual births per female in the cohort
        self.base_asfr = np.zeros(18)
        self.base_asfr[3:10] = [0.015, 0.065, 0.095, 0.080, 0.045, 0.015, 0.002] # Cohorts 3=15-19, ..., 9=45-49
        
        # Baseline age-gender specific mortality rates (ASMR)
        self.base_asmr = np.zeros((18, 2))
        # Males (gender = 0)
        self.base_asmr[:, 0] = [
            0.0015, 0.0003, 0.0004, 0.0012, 0.0018, 0.0022, 0.0028, 0.0035, 
            0.0045, 0.0060, 0.0090, 0.0140, 0.0220, 0.0350, 0.0550, 0.0900, 0.1400, 0.2200
        ]
        # Females (gender = 1)
        self.base_asmr[:, 1] = [
            0.0012, 0.0002, 0.0003, 0.0005, 0.0007, 0.0009, 0.0012, 0.0016, 
            0.0022, 0.0032, 0.0050, 0.0080, 0.0130, 0.0220, 0.0380, 0.0650, 0.1100, 0.1800
        ]
        
        # Baseline Active -> Disabled transition rates
        self.base_disability = np.zeros((18, 2))
        # Males
        self.base_disability[:, 0] = [
            0.0005, 0.0006, 0.0008, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 
            0.0055, 0.0075, 0.0110, 0.0160, 0.0230, 0.0320, 0.0450, 0.0650, 0.0900, 0.1300
        ]
        # Females
        self.base_disability[:, 1] = [
            0.0004, 0.0005, 0.0007, 0.0010, 0.0013, 0.0017, 0.0022, 0.0030, 
            0.0042, 0.0060, 0.0090, 0.0130, 0.0190, 0.0270, 0.0380, 0.0550, 0.0800, 0.1100
        ]

        self.share_unskilled = 0.50
        self.share_semiskilled = 0.35
        self.share_skilled = 0.15

        if distances is not None:
            self.distances = distances
        else:
            self.distances = self._init_distance_matrix()

    @property
    def pop(self):
        pop_dict = {}
        for r_idx, r in enumerate(self.regions):
            pop_dict[r] = {
                'Male': self.pop_array[r_idx, :, 0, :].sum(axis=1),
                'Female': self.pop_array[r_idx, :, 1, :].sum(axis=1)
            }
        return pop_dict

    @pop.setter
    def pop(self, value):
        if isinstance(value, dict):
            self.pop_array = np.zeros((self.R, 18, 2, 2))
            for r_idx, r in enumerate(self.regions):
                for g_idx, g in enumerate(['Male', 'Female']):
                    pop_data = value[r][g]
                    if len(pop_data) == 3:
                        pop_18 = self._interpolate_3_to_18(pop_data)
                    else:
                        pop_18 = np.array(pop_data, dtype=float)
                    self.pop_array[r_idx, :, g_idx, 0] = pop_18 * 0.96
                    self.pop_array[r_idx, :, g_idx, 1] = pop_18 * 0.04
        else:
            self.pop_array = value

    def _interpolate_3_to_18(self, pop_3):
        """
        Interpolates 3 aggregate cohorts (0-14, 15-64, 65+) into 18 five-year cohorts.
        """
        pop_18 = np.zeros(18)
        # Cohorts 0, 1, 2 (0-14 years): 3 cohorts
        pop_18[0:3] = pop_3[0] / 3.0
        # Cohorts 3 to 12 (15-64 years): 10 cohorts
        pop_18[3:13] = pop_3[1] / 10.0
        # Cohorts 13 to 17 (65+ years): 5 cohorts
        pop_18[13:18] = pop_3[2] / 5.0
        return pop_18

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

    def step(self, year, scenario_modifiers, abm=None):
        """
        Simulate one year demographic shift.
        If abm is provided, we perform the steps on the abm agent arrays and aggregate back.
        """
        repatriation_rate = scenario_modifiers.get('repatriation_rate', 0.04)
        mobilization_rate = scenario_modifiers.get('mobilization_rate', 0.02)
        labor_participation = scenario_modifiers.get('labor_participation', 0.70)
        brain_drain_rate = scenario_modifiers.get('brain_drain_rate', 0.005)
        fertility_modifier = scenario_modifiers.get('fertility_modifier', 1.0)
        
        # Scenario war mortality penalty (increases working age male mortality rate)
        war_mortality_mult = scenario_modifiers.get('war_mortality_mult', 1.0)

        if abm is not None:
            # ----------------------------------------------------
            # MICRO AGENT-BASED VECTORIZED DEMOGRAPHICS
            # ----------------------------------------------------
            N = abm.num_households
            N = abm.num_households
            active_mask = abm.agent_health != 2  # Not deceased
            
            # 1. Increment Age & Education transitions
            abm.agent_age[active_mask] += 1
            abm.agent_cohort = np.clip(abm.agent_age // 5, 0, 17)
            
            # Education transition model for agents reaching working age (cohort 3: age 18)
            # Unskilled (0) -> Semi-skilled (1) or Skilled (2)
            transition_age = 18
            at_transition = active_mask & (abm.agent_age == transition_age) & (abm.agent_labor == 0)
            num_at_trans = np.sum(at_transition)
            if num_at_trans > 0:
                edu_mult = scenario_modifiers.get('education_spending_mult', 1.0)
                p_skilled = np.clip(0.15 * edu_mult, 0.05, 0.40)
                p_semiskilled = np.clip(0.35 * edu_mult, 0.15, 0.50)
                p_unskilled = 1.0 - p_skilled - p_semiskilled
                
                new_labor = np.random.choice([0, 1, 2], size=num_at_trans, p=[p_unskilled, p_semiskilled, p_skilled])
                abm.agent_labor[at_transition] = new_labor
                
            # Adult retraining model for working-age active agents (cohort 3-12, age > 18)
            working_adults = active_mask & (abm.agent_age > transition_age) & (abm.agent_cohort >= 3) & (abm.agent_cohort <= 12) & (abm.agent_health == 0)
            
            # Unskilled -> Semi-skilled
            unskilled_adults = working_adults & (abm.agent_labor == 0)
            num_unskilled = np.sum(unskilled_adults)
            if num_unskilled > 0:
                edu_mult = scenario_modifiers.get('education_spending_mult', 1.0)
                p_retrain_u = np.clip(0.02 * edu_mult, 0.0, 0.10)
                draws = np.random.rand(num_unskilled)
                upgrade_mask = draws < p_retrain_u
                indices = np.where(unskilled_adults)[0]
                abm.agent_labor[indices[upgrade_mask]] = 1
                
            # Semi-skilled -> Skilled
            semiskilled_adults = working_adults & (abm.agent_labor == 1)
            num_semiskilled = np.sum(semiskilled_adults)
            if num_semiskilled > 0:
                edu_mult = scenario_modifiers.get('education_spending_mult', 1.0)
                p_retrain_m = np.clip(0.01 * edu_mult, 0.0, 0.05)
                draws = np.random.rand(num_semiskilled)
                upgrade_mask = draws < p_retrain_m
                indices = np.where(semiskilled_adults)[0]
                abm.agent_labor[indices[upgrade_mask]] = 2
            
            # 2. Health Morbidity Transitions (Active -> Disabled)
            disable_probs = self.base_disability[abm.agent_cohort, abm.agent_gender].copy()
            # Frontline region risk increases civilian disability
            for r_idx, r in enumerate(self.regions):
                r_mask = abm.agent_region == r_idx
                risk_lvl = scenario_modifiers.get('frontline_states', {}).get(r, 0)
                if risk_lvl == 1:
                    disable_probs[r_mask & active_mask] += 0.003
                elif risk_lvl == 2:
                    disable_probs[r_mask & active_mask] += 0.010
                    
            # Combat-specific disability for mobilized age groups (particularly males)
            war_intensity = scenario_modifiers.get('war_damage_intensity', 0.02)
            combat_age = (abm.agent_cohort >= 3) & (abm.agent_cohort <= 9)
            males = abm.agent_gender == 0
            combat_disable_prob = war_intensity * mobilization_rate * 2.5
            disable_probs[combat_age & males & active_mask] += combat_disable_prob

            disable_draw = np.random.rand(N)
            disable_mask = active_mask & (abm.agent_health == 0) & (disable_draw < disable_probs)
            abm.agent_health[disable_mask] = 1 # Active -> Disabled
            
            # 3. Mortality (Active and Disabled have different rates)
            mort_probs = self.base_asmr[abm.agent_cohort, abm.agent_gender].copy()
            
            # War mortality penalty for working age males (cohorts 3 to 12)
            working_age_male = (abm.agent_gender == 0) & (abm.agent_cohort >= 3) & (abm.agent_cohort <= 12)
            mort_probs[working_age_male & active_mask] *= war_mortality_mult
            
            # Disabled agents have higher mortality (3.0x multiplier, capped at 0.9)
            disabled_mask = abm.agent_health == 1
            mort_probs[disabled_mask] = np.clip(mort_probs[disabled_mask] * 3.0, 0.0, 0.9)
            
            # Frontline region direct war deaths
            for r_idx, r in enumerate(self.regions):
                r_mask = abm.agent_region == r_idx
                risk_lvl = scenario_modifiers.get('frontline_states', {}).get(r, 0)
                if risk_lvl == 1:
                    mort_probs[r_mask & active_mask] += 0.002
                elif risk_lvl == 2:
                    mort_probs[r_mask & active_mask] += 0.015

            death_draw = np.random.rand(N)
            death_mask = active_mask & (death_draw < mort_probs)
            
            # Set deceased status
            abm.agent_health[death_mask] = 2 # Deceased
            total_deaths = int(np.sum(death_mask))
            
            # 4. Fertility (Births)
            female_mask = (abm.agent_gender == 1) & (abm.agent_cohort >= 3) & (abm.agent_cohort <= 9)
            birth_probs = self.base_asfr[abm.agent_cohort] * fertility_modifier
            birth_draw = np.random.rand(N)
            birth_mask = active_mask & female_mask & (birth_draw < birth_probs)
            expected_births = int(np.sum(birth_mask))
            
            # Recycle deceased slots to spawn newborns
            deceased_slots = np.where(abm.agent_health == 2)[0]
            num_to_spawn = min(expected_births, len(deceased_slots))
            
            if num_to_spawn > 0:
                spawn_indices = deceased_slots[:num_to_spawn]
                mothers_regions = abm.agent_region[birth_mask][:num_to_spawn]
                
                abm.agent_region[spawn_indices] = mothers_regions
                abm.agent_age[spawn_indices] = 0
                abm.agent_cohort[spawn_indices] = 0
                abm.agent_gender[spawn_indices] = np.random.choice([0, 1], size=num_to_spawn, p=[0.51, 0.49])
                abm.agent_health[spawn_indices] = 0 # Active
                abm.agent_wealth[spawn_indices] = 0.0
                abm.agent_labor[spawn_indices] = 0 # Unskilled start
            
            total_births = num_to_spawn
            
            # 5. External Migration / Refugee Returns & Brain Drain
            # Brain drain: Active skilled labor flees the country (set to health 3)
            skilled_active = (abm.agent_labor == 2) & (abm.agent_health == 0) & active_mask
            brain_drain_draw = np.random.rand(N)
            flee_mask = skilled_active & (brain_drain_draw < brain_drain_rate)
            abm.agent_health[flee_mask] = 3 # Emigrated
            
            # Repatriation: transition from Emigrated (3) back to Active (0) in Ukraine
            emigrated_mask = abm.agent_health == 3
            num_emigrated = np.sum(emigrated_mask)
            if num_emigrated > 0:
                return_draw = np.random.rand(N)
                return_mask = emigrated_mask & (return_draw < repatriation_rate)
                num_returning = np.sum(return_mask)
                if num_returning > 0:
                    grp_per_capita = scenario_modifiers.get('grp_per_capita', {r: 1.0 for r in self.regions})
                    attractions = np.array([grp_per_capita[r] for r in self.regions])
                    attractions /= np.sum(attractions)
                    ref_regions = np.random.choice(self.R, size=num_returning, p=attractions)
                    
                    abm.agent_region[return_mask] = ref_regions
                    abm.agent_health[return_mask] = 0 # Return to Active status

            # Spawn new agents from pre-2026 external pool to fill deceased slots
            refugee_pool = scenario_modifiers.get('refugee_pool', 5.0e6)
            returned_refugees = int(refugee_pool * repatriation_rate * 0.10)
            
            deceased_slots = np.where(abm.agent_health == 2)[0]
            num_refugees_to_spawn = min(returned_refugees, len(deceased_slots))
            
            if num_refugees_to_spawn > 0:
                spawn_indices = deceased_slots[:num_refugees_to_spawn]
                grp_per_capita = scenario_modifiers.get('grp_per_capita', {r: 1.0 for r in self.regions})
                attractions = np.array([grp_per_capita[r] for r in self.regions])
                attractions /= np.sum(attractions)
                ref_regions = np.random.choice(self.R, size=num_refugees_to_spawn, p=attractions)
                
                ref_cohorts = np.random.choice(
                    np.arange(18), size=num_refugees_to_spawn,
                    p=[0.08,0.08,0.09, 0.08,0.08,0.08,0.08,0.07,0.06,0.05,0.05,0.05,0.05, 0.02,0.02,0.02,0.02,0.02]
                )
                
                abm.agent_region[spawn_indices] = ref_regions
                abm.agent_cohort[spawn_indices] = ref_cohorts
                abm.agent_age[spawn_indices] = ref_cohorts * 5 + np.random.randint(0, 5, size=num_refugees_to_spawn)
                abm.agent_gender[spawn_indices] = np.random.choice([0, 1], size=num_refugees_to_spawn, p=[0.45, 0.55])
                abm.agent_health[spawn_indices] = np.random.choice([0, 1], size=num_refugees_to_spawn, p=[0.96, 0.04])
                abm.agent_wealth[spawn_indices] = 10000.0
                abm.agent_labor[spawn_indices] = np.random.choice([0, 1, 2], size=num_refugees_to_spawn, p=[0.50, 0.35, 0.15])
            
            # Optimized Vectorized Demographics Aggregation using np.bincount (excludes health 2 and 3)
            living = abm.agent_health <= 1
            flat_indices = (abm.agent_region[living].astype(int) * 72 + 
                            abm.agent_cohort[living].astype(int) * 4 + 
                            abm.agent_gender[living].astype(int) * 2 + 
                            abm.agent_health[living].astype(int))
            counts = np.bincount(flat_indices, minlength=27 * 18 * 2 * 2)
            self.pop_array = counts.reshape((self.R, 18, 2, 2))
            
            # Optimized Vectorized Labor Supply Calculation using np.bincount
            real_people_per_agent = 10.0
            active_working = (abm.agent_cohort >= 3) & (abm.agent_cohort <= 12) & (abm.agent_health == 0)
            males_active = active_working & (abm.agent_gender == 0)
            females_active = active_working & (abm.agent_gender == 1)
            
            flat_labor_indices = abm.agent_region.astype(int) * 3 + abm.agent_labor.astype(int)
            
            male_counts = np.bincount(flat_labor_indices[males_active], minlength=81).reshape((27, 3))
            female_counts = np.bincount(flat_labor_indices[females_active], minlength=81).reshape((27, 3))
            
            # Calculate supply values
            supply_matrix = (male_counts * (1.0 - mobilization_rate) + female_counts) * labor_participation * real_people_per_agent
            
            labor_supply = {}
            for r_idx, r in enumerate(self.regions):
                labor_supply[r] = {
                    'unskilled': max(100.0, supply_matrix[r_idx, 0]),
                    'semi-skilled': max(100.0, supply_matrix[r_idx, 1]),
                    'skilled': max(100.0, supply_matrix[r_idx, 2])
                }
            
        else:
            # ----------------------------------------------------
            # MACRO AGGREGATE LESLIE TRANSITIONS (FALLBACK)
            # ----------------------------------------------------
            new_pop = np.zeros_like(self.pop_array)
            total_births = 0.0
            total_deaths = 0.0
            
            for r_idx in range(self.R):
                for g_idx in range(2):
                    for h_idx in range(2):
                        current = self.pop_array[r_idx, :, g_idx, h_idx]
                        
                        # Apply mortality
                        asmr = self.base_asmr[:, g_idx].copy()
                        if g_idx == 0: # male war mortality
                            asmr[3:13] *= war_mortality_mult
                        if h_idx == 1: # disabled mortality penalty
                            asmr = np.clip(asmr * 3.0, 0.0, 0.9)
                            
                        survivors = current * (1.0 - asmr)
                        total_deaths += np.sum(current * asmr)
                        
                        # Continuous aging: 20% move to next cohort
                        aging_out = survivors * 0.20
                        aging_out[17] = 0.0 # Cohort 17 (85+) stays in cohort 17
                        
                        new_pop[r_idx, 0, g_idx, h_idx] = survivors[0] - aging_out[0]
                        for c in range(1, 17):
                            new_pop[r_idx, c, g_idx, h_idx] = survivors[c] - aging_out[c] + aging_out[c-1]
                        new_pop[r_idx, 17, g_idx, h_idx] = survivors[17] + aging_out[16]
                        
                # Disability transitions
                active_pop = new_pop[r_idx, :, :, 0]
                dis_rates = self.base_disability.copy()
                # Conflict increases disability rate
                r_name = self.regions[r_idx]
                risk_lvl = scenario_modifiers.get('frontline_states', {}).get(r_name, 0)
                if risk_lvl == 1:
                    dis_rates += 0.005
                elif risk_lvl == 2:
                    dis_rates += 0.020
                    
                new_disabled = active_pop * dis_rates
                new_pop[r_idx, :, :, 0] -= new_disabled
                new_pop[r_idx, :, :, 1] += new_disabled
                
                # Births (females in cohorts 3-9)
                females_fertile = self.pop_array[r_idx, 3:10, 1, :].sum(axis=1) # active + disabled females
                births = np.sum(females_fertile * self.base_asfr[3:10] * fertility_modifier)
                total_births += births
                
                # Add newborns to cohort 0, health 0 (Active), 51% male, 49% female
                new_pop[r_idx, 0, 0, 0] += births * 0.51
                new_pop[r_idx, 0, 1, 0] += births * 0.49
                
            # Internal migration for aggregate pop (based on GRP attractions)
            grp_per_capita = scenario_modifiers.get('grp_per_capita', {r: 1.0 for r in self.regions})
            attractions = np.array([grp_per_capita[r] for r in self.regions])
            migration_outflow_rate = 0.015
            
            migrants = new_pop * migration_outflow_rate
            new_pop -= migrants
            
            for i in range(self.R):
                dists = self.distances[i]
                weights = (attractions ** self.gravity['attraction']) / (dists ** self.gravity['distance_decay'])
                weights[i] = 0.0
                sum_w = np.sum(weights)
                if sum_w > 0:
                    weights /= sum_w
                    
                for j in range(self.R):
                    new_pop[j] += migrants[i] * weights[j]
                    
            # Return refugees & Brain drain
            refugee_pool = scenario_modifiers.get('refugee_pool', 5.0e6)
            returned_refugees = refugee_pool * repatriation_rate
            
            # Brain drain
            for r_idx in range(self.R):
                for g_idx in range(2):
                    # Loss of skilled working age population (approx cohort 3-12 active)
                    loss = new_pop[r_idx, 3:13, g_idx, 0] * brain_drain_rate
                    new_pop[r_idx, 3:13, g_idx, 0] -= loss
                    total_deaths += np.sum(loss)
                    
                    # Returned refugees (distributed by GRP attraction)
                    region_share = attractions[r_idx] / np.sum(attractions)
                    returns = returned_refugees * region_share
                    
                    # Distribute by cohort skewing (children 25%, working 65%, elderly 10%)
                    returns_cohorts = self._interpolate_3_to_18([returns * 0.25, returns * 0.65, returns * 0.10])
                    new_pop[r_idx, :, g_idx, 0] += returns_cohorts * (0.45 if g_idx == 0 else 0.55)
            
            self.pop_array = new_pop
            
            # Calculate labor supply for compatibility with tests
            # Dynamic macro retraining update
            edu_mult = scenario_modifiers.get('education_spending_mult', 1.0)
            retrain_u_to_m = 0.02 * edu_mult
            retrain_m_to_s = 0.01 * edu_mult
            
            # Apply retraining shifts to the macro shares
            flow_u_to_m = self.share_unskilled * retrain_u_to_m
            flow_m_to_s = self.share_semiskilled * retrain_m_to_s
            
            self.share_unskilled = max(0.10, self.share_unskilled - flow_u_to_m)
            self.share_semiskilled = max(0.10, self.share_semiskilled + flow_u_to_m - flow_m_to_s)
            self.share_skilled = max(0.05, self.share_skilled + flow_m_to_s)
            
            labor_supply = {}
            for r_idx, r in enumerate(self.regions):
                active_working = new_pop[r_idx, 3:13, :, 0] # cohort 3-12, Active
                males = active_working[:, 0] * (1.0 - mobilization_rate)
                females = active_working[:, 1]
                labor_supply[r] = {
                    'unskilled': max(100.0, np.sum(males + females) * self.share_unskilled * labor_participation),
                    'semi-skilled': max(100.0, np.sum(males + females) * self.share_semiskilled * labor_participation),
                    'skilled': max(100.0, np.sum(males + females) * self.share_skilled * labor_participation)
                }

        total_pop_sum = float(np.sum(self.pop_array))
        refugee_pool = scenario_modifiers.get('refugee_pool', 5.0e6)
        returned_ref_actual = refugee_pool * repatriation_rate
        
        # Format the self.pop output for test_engine compatibility (it expects a dict of region -> {'Male': array, 'Female': array})
        # Let's construct a compatible pop_dict property dynamically on step returns or just keep a dict version in self.pop_dict
        self.pop_dict = {}
        for r_idx, r in enumerate(self.regions):
            self.pop_dict[r] = {
                'Male': self.pop_array[r_idx, :, 0, :].sum(axis=1),
                'Female': self.pop_array[r_idx, :, 1, :].sum(axis=1)
            }
            
        return {
            'population': self.pop_dict,
            'labor_supply': labor_supply,
            'total_pop': total_pop_sum,
            'births': float(total_births if abm is None else expected_births),
            'deaths': float(total_deaths),
            'refugees_remaining': max(0.0, refugee_pool - returned_ref_actual)
        }
