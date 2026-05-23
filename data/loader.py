import numpy as np

# List of 27 regions (oblasts, Autonomous Republic of Crimea, Kyiv, Sevastopol)
REGIONS = [
    'Cherkasy', 'Chernihiv', 'Chernivtsi', 'Dnipro', 'Donetsk', 'Ivano-Frankivsk',
    'Kharkiv', 'Kherson', 'Khmelnytskyi', 'Kyiv_Oblast', 'Kirovohrad', 'Luhansk',
    'Lviv', 'Mykolaiv', 'Odesa', 'Poltava', 'Rivne', 'Sumy', 'Ternopil', 'Vinnytsia',
    'Volyn', 'Zakarpattia', 'Zaporizhzhia', 'Zhytomyr', 'Crimea', 'Kyiv_City', 'Sevastopol'
]

# List of 15 sectors
SECTORS = [
    'Agriculture', 'ConsumerGoods', 'Metallurgy', 'Energy', 'IT',
    'Machinery', 'MilitaryIndustrial', 'Construction', 'Transport',
    'Retail', 'Finance', 'Healthcare', 'PublicAdmin', 'Chemicals', 'Tourism'
]

def load_base_technical_coefficients():
    """
    Returns direct requirements technical coefficients matrix for the 15 sectors.
    Represents how much input from sector J (key) is needed per unit of output in sector I (nested key).
    """
    return {
        'Agriculture': {
            'Energy': 0.08, 'Chemicals': 0.10, 'Transport': 0.06, 'Finance': 0.02, 'Agriculture': 0.08
        },
        'ConsumerGoods': {
            'Agriculture': 0.22, 'Chemicals': 0.05, 'Energy': 0.06, 'Transport': 0.07, 'Finance': 0.03
        },
        'Metallurgy': {
            'Energy': 0.16, 'Transport': 0.10, 'Chemicals': 0.04, 'Finance': 0.03, 'Metallurgy': 0.12
        },
        'Energy': {
            'Metallurgy': 0.04, 'Transport': 0.03, 'Chemicals': 0.05, 'Finance': 0.02, 'Energy': 0.08
        },
        'IT': {
            'Energy': 0.03, 'Finance': 0.06, 'ConsumerGoods': 0.02, 'IT': 0.05
        },
        'Machinery': {
            'Metallurgy': 0.20, 'Energy': 0.07, 'Chemicals': 0.04, 'Transport': 0.05, 'Finance': 0.04
        },
        'MilitaryIndustrial': {
            'Metallurgy': 0.15, 'Machinery': 0.18, 'Chemicals': 0.08, 'Energy': 0.09, 'Transport': 0.06, 'IT': 0.08
        },
        'Construction': {
            'Metallurgy': 0.12, 'Energy': 0.05, 'Chemicals': 0.06, 'Transport': 0.08, 'Finance': 0.03
        },
        'Transport': {
            'Energy': 0.18, 'Machinery': 0.08, 'Finance': 0.04, 'Transport': 0.05
        },
        'Retail': {
            'ConsumerGoods': 0.30, 'Energy': 0.04, 'Transport': 0.06, 'Finance': 0.05
        },
        'Finance': {
            'IT': 0.07, 'Energy': 0.02, 'Finance': 0.10
        },
        'Healthcare': {
            'Chemicals': 0.15, 'Energy': 0.04, 'Finance': 0.02
        },
        'PublicAdmin': {
            'Energy': 0.05, 'Transport': 0.04, 'IT': 0.06, 'Finance': 0.02
        },
        'Chemicals': {
            'Energy': 0.12, 'Chemicals': 0.10, 'Transport': 0.05, 'Finance': 0.03
        },
        'Tourism': {
            'ConsumerGoods': 0.10, 'Energy': 0.06, 'Transport': 0.12, 'Retail': 0.05
        }
    }

def generate_baseline_data():
    """
    Procedurally synthesizes the baseline state for Ukraine's economy in 2026.
    Ensures correct sizing, consistency, and realistic distribution.
    """
    np.random.seed(42) # Deterministic baseline generation
    
    # 1. Demographic Distribution (Total ~34 million)
    # Kyiv, Dnipro, Lviv, Kharkiv, Odesa are major population hubs
    # Frontline/occupied regions (Donetsk, Luhansk, Kherson, Crimea, Sevastopol) are initialized with depleted sizes.
    reg_pop_weights = {
        'Kyiv_City': 2.9, 'Dnipro': 2.8, 'Lviv': 2.5, 'Kharkiv': 2.3, 'Odesa': 2.1,
        'Kyiv_Oblast': 1.8, 'Vinnytsia': 1.5, 'Poltava': 1.3, 'Zaporizhzhia': 1.2,
        'Ivano-Frankivsk': 1.3, 'Khmelnytskyi': 1.2, 'Cherkasy': 1.1, 'Zhytomyr': 1.1,
        'Donetsk': 1.0, 'Zakarpattia': 1.2, 'Rivne': 1.1, 'Sumy': 0.9, 'Ternopil': 1.0,
        'Volyn': 1.0, 'Chernihiv': 0.9, 'Kirovohrad': 0.8, 'Mykolaiv': 0.9,
        'Chernivtsi': 0.9, 'Luhansk': 0.5, 'Kherson': 0.5, 'Crimea': 1.4, 'Sevastopol': 0.3
    }
    
    total_weight = sum(reg_pop_weights.values())
    scale_factor = 34.0e6 / total_weight # Scale to 34 million total population
    
    initial_pop = {}
    fertility_rates = {}
    mortality_rates = {}
    
    for r in REGIONS:
        weight = reg_pop_weights.get(r, 1.0)
        pop_size = weight * scale_factor
        
        # Gender split: ~47% Male, ~53% Female (skewed due to migration/war/longevity)
        m_pop = pop_size * 0.47
        f_pop = pop_size * 0.53
        
        # Cohorts: 0-14, 15-64, 65+
        # Male age distribution: ~16% children, ~67% working, ~17% elderly
        initial_pop[r] = {
            'Male': np.array([m_pop * 0.16, m_pop * 0.67, m_pop * 0.17]),
            'Female': np.array([f_pop * 0.14, f_pop * 0.64, f_pop * 0.22]) # skew towards elderly females
        }
        
        # Fertility: lower in cities, slightly higher in rural western regions
        # Range: 0.012 to 0.017 births per female in 15-64 cohort per year
        if r in ['Kyiv_City', 'Kharkiv', 'Lviv', 'Odesa']:
            fertility_rates[r] = 0.013
        elif r in ['Zakarpattia', 'Volyn', 'Rivne', 'Ivano-Frankivsk']:
            fertility_rates[r] = 0.018
        else:
            fertility_rates[r] = 0.015
            
        # Mortality rates: cohort-specific death probability per year
        mortality_rates[r] = {
            'Male': np.array([0.0012, 0.0085, 0.062]), # Higher working-age male mortality
            'Female': np.array([0.0009, 0.0035, 0.042])
        }

    # Gravity coefficients for internal migration
    migration_gravity = {
        'attraction': 0.8,
        'distance_decay': 1.2
    }

    # 2. Capital Stock & GRP Distribution (Total GDP ~ 8.0 trillion UAH, Capital Stock ~ 22.0 trillion UAH)
    # GRP weights are aligned with population hubs, but Kyiv City has much higher productivity per capita.
    reg_grp_weights = {
        'Kyiv_City': 0.24, 'Dnipro': 0.10, 'Lviv': 0.08, 'Kharkiv': 0.06, 'Odesa': 0.06,
        'Kyiv_Oblast': 0.05, 'Poltava': 0.04, 'Zaporizhzhia': 0.03, 'Vinnytsia': 0.03,
        'Ivano-Frankivsk': 0.03, 'Khmelnytskyi': 0.025, 'Cherkasy': 0.025, 'Zhytomyr': 0.02,
        'Donetsk': 0.015, 'Zakarpattia': 0.02, 'Rivne': 0.02, 'Sumy': 0.018, 'Ternopil': 0.018,
        'Volyn': 0.018, 'Chernihiv': 0.015, 'Kirovohrad': 0.015, 'Mykolaiv': 0.015,
        'Chernivtsi': 0.015, 'Luhansk': 0.005, 'Kherson': 0.005, 'Crimea': 0.015, 'Sevastopol': 0.005
    }
    
    total_grp_weight = sum(reg_grp_weights.values())
    total_capital = 22.0e12 # 22 trillion UAH total capital stock
    
    initial_capital = {}
    initial_tfp = {}
    energy_utilization = {}
    
    # Sector distributions within GRP (varies by region type)
    # e.g., Agriculture is high in Poltava/Vinnytsia, IT in Kyiv/Lviv, Metallurgy in Dnipro/Zaporizhzhia
    for r in REGIONS:
        initial_capital[r] = {}
        initial_tfp[r] = {}
        energy_utilization[r] = {}
        
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_total_cap = total_capital * reg_weight
        
        # Determine sector allocation weights for this region
        sector_weights = {s: 1.0 for s in SECTORS}
        
        # Customize sectoral footprints per region
        if r == 'Kyiv_City':
            sector_weights['IT'] = 5.0
            sector_weights['Finance'] = 4.0
            sector_weights['Retail'] = 3.0
            sector_weights['PublicAdmin'] = 3.0
            sector_weights['Agriculture'] = 0.05
            sector_weights['Metallurgy'] = 0.05
        elif r in ['Dnipro', 'Zaporizhzhia', 'Donetsk', 'Luhansk']:
            sector_weights['Metallurgy'] = 5.0
            sector_weights['Machinery'] = 3.0
            sector_weights['Energy'] = 3.0
            sector_weights['IT'] = 0.5
        elif r in ['Poltava', 'Vinnytsia', 'Cherkasy', 'Kirovohrad', 'Khmelnytskyi']:
            sector_weights['Agriculture'] = 4.0
            sector_weights['ConsumerGoods'] = 2.0
            sector_weights['Metallurgy'] = 0.1
        elif r in ['Lviv', 'Kharkiv']:
            sector_weights['IT'] = 3.0
            sector_weights['Machinery'] = 2.0
            sector_weights['Tourism'] = 2.5
            sector_weights['Agriculture'] = 0.5
        elif r in ['Odesa', 'Mykolaiv', 'Kherson']:
            sector_weights['Transport'] = 4.0 # Ports & logistics
            sector_weights['Agriculture'] = 2.0
            sector_weights['Tourism'] = 2.0
            sector_weights['Metallurgy'] = 0.3
            
        sum_sw = sum(sector_weights.values())
        
        for s in SECTORS:
            share = sector_weights[s] / sum_sw
            initial_capital[r][s] = reg_total_cap * share
            
            # TFP baseline: Kyiv has highest efficiency, frontline/occupied have lowest due to damaged assets
            base_tfp = 1.0
            if r in ['Kyiv_City']:
                base_tfp = 1.4
            elif r in ['Lviv', 'Dnipro']:
                base_tfp = 1.1
            elif r in ['Donetsk', 'Luhansk', 'Kherson', 'Crimea', 'Sevastopol']:
                base_tfp = 0.4 # Heavily damaged/unreformed structures
                
            # Sector adjustments to keep GDP order of magnitude correct
            sector_tfp_mult = {
                'Agriculture': 0.18, 'ConsumerGoods': 0.14, 'Metallurgy': 0.15, 'Energy': 0.22,
                'IT': 0.45, 'Machinery': 0.16, 'MilitaryIndustrial': 0.20, 'Construction': 0.12,
                'Transport': 0.16, 'Retail': 0.14, 'Finance': 0.25, 'Healthcare': 0.10,
                'PublicAdmin': 0.08, 'Chemicals': 0.15, 'Tourism': 0.12
            }
            
            initial_tfp[r][s] = base_tfp * sector_tfp_mult[s] * 18000.0
            
            # Energy grid constraints: 2026 starts with grid challenges
            # Frontline regions and energy hubs are damaged
            if r in ['Donetsk', 'Luhansk', 'Kharkiv', 'Kherson', 'Zaporizhzhia']:
                energy_utilization[r][s] = 0.65 # 35% energy deficit limit
            elif r in ['Kyiv_City', 'Kyiv_Oblast', 'Odesa']:
                energy_utilization[r][s] = 0.80 # 20% energy deficit limit
            else:
                energy_utilization[r][s] = 0.92 # 8% deficit

    # 3. Baseline prices (normalized around 1.0)
    prices = {r: {s: 1.0 for s in SECTORS} for r in REGIONS}
    
    # 4. Target Final Demand (Consumer, Government, Export)
    # Approx 60% of GRP is final demand
    target_final_demand = {}
    for r in REGIONS:
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_total_demand_uah = 5.2e12 * reg_weight # 5.2 trillion UAH total final demand
        
        sector_demand_weights = {
            'Agriculture': 0.08, 'ConsumerGoods': 0.25, 'Metallurgy': 0.05, 'Energy': 0.06,
            'IT': 0.08, 'Machinery': 0.05, 'MilitaryIndustrial': 0.08, 'Construction': 0.08,
            'Transport': 0.06, 'Retail': 0.05, 'Finance': 0.04, 'Healthcare': 0.05,
            'PublicAdmin': 0.04, 'Chemicals': 0.02, 'Tourism': 0.01
        }
        
        for s in SECTORS:
            share = sector_demand_weights[s]
            target_final_demand[(r, s)] = reg_total_demand_uah * share

    # 5. ABM / CGE Parameters
    budget_shares = {
        'Agriculture': 0.08, 'ConsumerGoods': 0.25, 'Metallurgy': 0.05, 'Energy': 0.06,
        'IT': 0.08, 'Machinery': 0.05, 'MilitaryIndustrial': 0.08, 'Construction': 0.08,
        'Transport': 0.06, 'Retail': 0.05, 'Finance': 0.04, 'Healthcare': 0.05,
        'PublicAdmin': 0.04, 'Chemicals': 0.02, 'Tourism': 0.01
    }
    
    # Normalize budget shares
    sum_bs = sum(budget_shares.values())
    budget_shares = {k: v / sum_bs for k, v in budget_shares.items()}
    
    subsistence_demands = {}
    for r in REGIONS:
        subsistence_demands[r] = {}
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_pop = initial_pop[r]['Male'].sum() + initial_pop[r]['Female'].sum()
        
        # Total consumer demand for this region
        reg_cons_uah = 3.0e12 * reg_weight
        
        for s in SECTORS:
            # Subsistence is 40% of standard budget, divided by population to get per-capita units
            share = budget_shares[s]
            subsistence_demands[r][s] = (0.40 * reg_cons_uah * share) / max(1e-1, reg_pop)
            
    wages_by_type = {}
    for r in REGIONS:
        mult = 1.5 if r == 'Kyiv_City' else (1.1 if r in ['Lviv', 'Dnipro'] else 0.8)
        wages_by_type[r] = {
            'skilled': 300000.0 * mult,
            'unskilled': 120000.0 * mult
        }

    return {
        'regions': REGIONS,
        'sectors': SECTORS,
        'initial_pop': initial_pop,
        'fertility_rates': fertility_rates,
        'mortality_rates': mortality_rates,
        'migration_gravity': migration_gravity,
        'initial_capital': initial_capital,
        'initial_tfp': initial_tfp,
        'energy_utilization': energy_utilization,
        'prices': prices,
        'target_final_demand': target_final_demand,
        'base_tech_coefficients': load_base_technical_coefficients(),
        'budget_shares': budget_shares,
        'subsistence_demands': subsistence_demands,
        'wages_by_type': wages_by_type
    }
