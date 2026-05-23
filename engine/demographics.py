import numpy as np
import pandas as pd

class DemographicEngine:
    """
    Cohort-Component demographic model for 27 Ukrainian regions.
    Tracks population across 3 age groups (0-14, 15-64, 65+) and 2 genders (Male, Female).
    Integrates internal regional migration and international refugee dynamics.
    """
    def __init__(self, regions, initial_pop, fertility_rates, mortality_rates, migration_gravity_coeffs):
        """
        initial_pop: dict of region -> {'Male': [c0, c1, c2], 'Female': [c0, c1, c2]}
        fertility_rates: dict of region -> rate (births per female aged 15-64)
        mortality_rates: dict of region -> {'Male': [m0, m1, m2], 'Female': [m0, m1, m2]}
        migration_gravity_coeffs: dict with keys 'attraction', 'distance_decay'
        """
        self.regions = regions
        self.pop = {r: {g: np.array(initial_pop[r][g], dtype=float) for g in ['Male', 'Female']} for r in regions}
        self.fertility_rates = fertility_rates
        self.mortality_rates = mortality_rates
        self.gravity = migration_gravity_coeffs
        
        # Approximate distance matrix between region centers (in km)
        # Normalized for gravity model calculation
        self.distances = self._init_distance_matrix()

    def _init_distance_matrix(self):
        n = len(self.regions)
        dist = np.zeros((n, n))
        # Simple synthetic distance matrix based on alphabetical order or regional clusters
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist[i, j] = 1.0
                else:
                    # Representative distances (e.g., Lviv to Kyiv ~540km, etc.)
                    # Procedural proxy: distance based on index difference
                    dist[i, j] = 100.0 + abs(i - j) * 35.0
        return dist

    def step(self, year, scenario_modifiers):
        """
        Simulate one year demographic shift.
        Returns:
            - dict of current populations
            - total labor supply per region
            - deaths and births summary
        """
        repatriation_rate = scenario_modifiers.get('repatriation_rate', 0.05) # Rate of returning refugees
        mobilization_rate = scenario_modifiers.get('mobilization_rate', 0.02) # Percentage of 15-64 Males mobilized (not in civilian labor)
        labor_participation = scenario_modifiers.get('labor_participation', 0.70)
        brain_drain_rate = scenario_modifiers.get('brain_drain_rate', 0.005)
        
        new_pop = {r: {'Male': np.zeros(3), 'Female': np.zeros(3)} for r in self.regions}
        total_births = 0
        total_deaths = 0
        
        # 1. Natural Growth (Births, Deaths, Aging)
        for r in self.regions:
            for g in ['Male', 'Female']:
                current = self.pop[r][g]
                m = self.mortality_rates[r][g]
                
                # Apply mortality to cohorts
                survivors = current * (1.0 - m)
                total_deaths += np.sum(current * m)
                
                # Cohort aging dynamics (yearly transition rate: 1/15 for 0-14, 1/50 for 15-64)
                aging_0_to_15 = survivors[0] / 15.0
                aging_15_to_65 = survivors[1] / 50.0
                
                new_pop[r][g][0] = survivors[0] - aging_0_to_15
                new_pop[r][g][1] = survivors[1] + aging_0_to_15 - aging_15_to_65
                new_pop[r][g][2] = survivors[2] + aging_15_to_65
                
            # Births (based on females in cohort 1: 15-64)
            females_15_64 = self.pop[r]['Female'][1]
            births = females_15_64 * self.fertility_rates[r] * scenario_modifiers.get('fertility_modifier', 1.0)
            total_births += births
            
            # 50/50 gender ratio at birth
            new_pop[r]['Male'][0] += births * 0.51
            new_pop[r]['Female'][0] += births * 0.49

        # 2. Internal Gravity Migration
        # Migrants move based on GRP per capita differences and distance
        grp_per_capita = scenario_modifiers.get('grp_per_capita', {r: 1.0 for r in self.regions})
        migration_outflow_rate = 0.015 # 1.5% baseline internal movement per year
        
        internal_migration_flows = {r: {'Male': np.zeros(3), 'Female': np.zeros(3)} for r in self.regions}
        
        # Calculate attraction index for each region
        attractions = np.array([grp_per_capita[r] for r in self.regions])
        
        for i, r_from in enumerate(self.regions):
            for g in ['Male', 'Female']:
                # Only age group 1 (15-64) and group 0 (children moving with parents) migrate significantly
                migrants_0 = new_pop[r_from][g][0] * migration_outflow_rate
                migrants_1 = new_pop[r_from][g][1] * migration_outflow_rate
                
                new_pop[r_from][g][0] -= migrants_0
                new_pop[r_from][g][1] -= migrants_1
                
                # Distribute migrants to other regions using gravity model
                dists = self.distances[i]
                weights = (attractions ** self.gravity['attraction']) / (dists ** self.gravity['distance_decay'])
                weights[i] = 0.0 # No self-migration
                sum_w = np.sum(weights)
                if sum_w > 0:
                    weights /= sum_w
                    
                for j, r_to in enumerate(self.regions):
                    internal_migration_flows[r_to][g][0] += migrants_0 * weights[j]
                    internal_migration_flows[r_to][g][1] += migrants_1 * weights[j]
                    
        # Apply internal migration flows
        for r in self.regions:
            for g in ['Male', 'Female']:
                new_pop[r][g] += internal_migration_flows[r][g]

        # 3. International Migration (Refugees returning or brain drain)
        refugee_pool = scenario_modifiers.get('refugee_pool', 5000000.0) # Total Ukrainian refugees abroad
        returned_refugees_this_year = refugee_pool * repatriation_rate
        
        # Brain drain outflow (mainly from age group 1)
        for r in self.regions:
            for g in ['Male', 'Female']:
                # Brain drain loss
                loss = new_pop[r][g][1] * brain_drain_rate
                new_pop[r][g][1] -= loss
                
                # Returned refugees distribution (distributed according to regional GRP attraction)
                # Weighted distribution
                region_idx = self.regions.index(r)
                attraction_share = attractions[region_idx] / sum(attractions)
                
                returns_0 = returned_refugees_this_year * attraction_share * 0.25 # 25% children
                returns_1 = returned_refugees_this_year * attraction_share * 0.65 # 65% working age
                returns_2 = returned_refugees_this_year * attraction_share * 0.10 # 10% elderly
                
                new_pop[r][g][0] += returns_0 * (0.51 if g == 'Male' else 0.49)
                new_pop[r][g][1] += returns_1 * (0.45 if g == 'Male' else 0.55) # Refugees skewed towards females
                new_pop[r][g][2] += returns_2 * (0.40 if g == 'Male' else 0.60)

        # Update core populations
        self.pop = new_pop
        
        # 4. Calculate Labor Supply
        # Labor supply = Population (15-64) * labor_participation * (1 - mobilization_rate [for males])
        labor_supply = {}
        for r in self.regions:
            males_15_64 = self.pop[r]['Male'][1]
            females_15_64 = self.pop[r]['Female'][1]
            
            avail_males = males_15_64 * (1.0 - mobilization_rate)
            avail_females = females_15_64 # Assuming female mobilization is negligible in active labor force
            
            labor_supply[r] = (avail_males + avail_females) * labor_participation

        total_pop_sum = sum(np.sum(self.pop[r]['Male'] + self.pop[r]['Female']) for r in self.regions)
        
        return {
            'population': self.pop,
            'labor_supply': labor_supply,
            'total_pop': total_pop_sum,
            'births': total_births,
            'deaths': total_deaths,
            'refugees_remaining': max(0.0, refugee_pool - returned_refugees_this_year)
        }
