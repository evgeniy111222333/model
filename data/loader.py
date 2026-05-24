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

# Geographic coordinates (latitude, longitude) of capital cities for the 27 regions
COORDINATES = {
    'Cherkasy': (49.4444, 32.0598),
    'Chernihiv': (51.4982, 31.2893),
    'Chernivtsi': (48.2908, 25.9345),
    'Dnipro': (48.4656, 35.0353),
    'Donetsk': (48.0159, 37.8028),
    'Ivano-Frankivsk': (48.9215, 24.7097),
    'Kharkiv': (49.9935, 36.2304),
    'Kherson': (46.6354, 32.6169),
    'Khmelnytskyi': (49.4230, 26.9871),
    'Kyiv_Oblast': (50.4501, 30.5234), # Sourced around Kyiv city
    'Kirovohrad': (48.5079, 32.2623),
    'Luhansk': (48.5740, 39.3078),
    'Lviv': (49.8397, 24.0297),
    'Mykolaiv': (46.9750, 31.9946),
    'Odesa': (46.4825, 30.7233),
    'Poltava': (49.5883, 34.5514),
    'Rivne': (50.6199, 26.2516),
    'Sumy': (50.9077, 34.7981),
    'Ternopil': (49.5535, 25.5948),
    'Vinnytsia': (49.2331, 28.4682),
    'Volyn': (50.7472, 25.3254),
    'Zakarpattia': (48.6208, 22.2879),
    'Zaporizhzhia': (47.8388, 35.1396),
    'Zhytomyr': (50.2547, 28.6587),
    'Crimea': (44.9521, 34.1024),
    'Kyiv_City': (50.4501, 30.5234),
    'Sevastopol': (44.6166, 33.5254)
}

def load_base_technical_coefficients():
    """
    Returns direct requirements technical coefficients matrix for the 15 sectors.
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

def calculate_geographic_distances():
    """
    Computes real road distances (in km) between the 27 oblast centers using 
    geodesic haversine distance corrected by a road curvature multiplier of 1.3.
    """
    n = len(REGIONS)
    dist = np.zeros((n, n))
    for i, r1 in enumerate(REGIONS):
        for j, r2 in enumerate(REGIONS):
            if i == j:
                dist[i, j] = 1.0 # home/self distance proxy
            else:
                lat1, lon1 = COORDINATES[r1]
                lat2, lon2 = COORDINATES[r2]
                
                # Haversine formula
                R = 6371.0 # Earth radius in km
                phi1 = np.radians(lat1)
                phi2 = np.radians(lat2)
                dphi = np.radians(lat2 - lat1)
                dlambda = np.radians(lon2 - lon1)
                
                a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
                c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                dist[i, j] = R * c * 1.3 # 30% curvature adjustment for real road detours
    return dist

def generate_baseline_data():
    """
    Procedurally synthesizes the baseline state for Ukraine's economy in 2026.
    Uses 18 five-year cohorts, real coordinates, and 3 labor types.
    """
    np.random.seed(42)
    
    # 1. Demographic Distribution (Total ~34 million)
    reg_pop_weights = {
        'Kyiv_City': 2.9, 'Dnipro': 2.8, 'Lviv': 2.5, 'Kharkiv': 2.3, 'Odesa': 2.1,
        'Kyiv_Oblast': 1.8, 'Vinnytsia': 1.5, 'Poltava': 1.3, 'Zaporizhzhia': 1.2,
        'Ivano-Frankivsk': 1.3, 'Khmelnytskyi': 1.2, 'Cherkasy': 1.1, 'Zhytomyr': 1.1,
        'Donetsk': 1.0, 'Zakarpattia': 1.2, 'Rivne': 1.1, 'Sumy': 0.9, 'Ternopil': 1.0,
        'Volyn': 1.0, 'Chernihiv': 0.9, 'Kirovohrad': 0.8, 'Mykolaiv': 0.9,
        'Chernivtsi': 0.9, 'Luhansk': 0.5, 'Kherson': 0.5, 'Crimea': 1.4, 'Sevastopol': 0.3
    }
    
    total_weight = sum(reg_pop_weights.values())
    scale_factor = 34.0e6 / total_weight
    
    initial_pop = {}
    fertility_rates = {}
    mortality_rates = {}
    
    # Realistic age-specific structure shares (18 cohorts)
    # Reflects declining birth rate, aging population, and war gender imbalance
    cohort_shares_male = np.array([
        0.052, 0.051, 0.050, 0.055, 0.062, 0.068, 0.072, 0.075, 0.072, 
        0.068, 0.062, 0.058, 0.052, 0.048, 0.040, 0.030, 0.020, 0.015
    ])
    cohort_shares_female = np.array([
        0.048, 0.047, 0.046, 0.050, 0.058, 0.064, 0.068, 0.072, 0.070, 
        0.068, 0.065, 0.062, 0.058, 0.055, 0.050, 0.042, 0.032, 0.025
    ])
    
    # Normalize shares
    cohort_shares_male /= np.sum(cohort_shares_male)
    cohort_shares_female /= np.sum(cohort_shares_female)
    
    for r in REGIONS:
        weight = reg_pop_weights.get(r, 1.0)
        pop_size = weight * scale_factor
        
        m_pop = pop_size * 0.47
        f_pop = pop_size * 0.53
        
        initial_pop[r] = {
            'Male': m_pop * cohort_shares_male,
            'Female': f_pop * cohort_shares_female
        }
        
        # Fertility modifier
        if r in ['Kyiv_City', 'Kharkiv', 'Lviv', 'Odesa']:
            fertility_rates[r] = 0.85
        elif r in ['Zakarpattia', 'Volyn', 'Rivne', 'Ivano-Frankivsk']:
            fertility_rates[r] = 1.15
        else:
            fertility_rates[r] = 1.0
            
        # Baseline mortality (18 cohorts)
        mortality_rates[r] = {
            'Male': np.array([
                0.0015, 0.0003, 0.0004, 0.0012, 0.0018, 0.0022, 0.0028, 0.0035, 
                0.0045, 0.0060, 0.0090, 0.0140, 0.0220, 0.0350, 0.0550, 0.0900, 0.1400, 0.2200
            ]),
            'Female': np.array([
                0.0012, 0.0002, 0.0003, 0.0005, 0.0007, 0.0009, 0.0012, 0.0016, 
                0.0022, 0.0032, 0.0050, 0.0080, 0.0130, 0.0220, 0.0380, 0.0650, 0.1100, 0.1800
            ])
        }

    migration_gravity = {
        'attraction': 0.8,
        'distance_decay': 1.2
    }

    # 2. Capital Stock & GRP (Total GDP ~ 8.0 trillion UAH, Capital ~ 22.0 trillion UAH)
    reg_grp_weights = {
        'Kyiv_City': 0.24, 'Dnipro': 0.10, 'Lviv': 0.08, 'Kharkiv': 0.06, 'Odesa': 0.06,
        'Kyiv_Oblast': 0.05, 'Poltava': 0.04, 'Zaporizhzhia': 0.03, 'Vinnytsia': 0.03,
        'Ivano-Frankivsk': 0.03, 'Khmelnytskyi': 0.025, 'Cherkasy': 0.025, 'Zhytomyr': 0.02,
        'Donetsk': 0.015, 'Zakarpattia': 0.02, 'Rivne': 0.02, 'Sumy': 0.018, 'Ternopil': 0.018,
        'Volyn': 0.018, 'Chernihiv': 0.015, 'Kirovohrad': 0.015, 'Mykolaiv': 0.015,
        'Chernivtsi': 0.015, 'Luhansk': 0.005, 'Kherson': 0.005, 'Crimea': 0.015, 'Sevastopol': 0.005
    }
    
    total_grp_weight = sum(reg_grp_weights.values())
    total_capital = 22.0e12
    
    initial_capital = {}
    initial_tfp = {}
    energy_utilization = {}
    
    for r in REGIONS:
        initial_capital[r] = {}
        initial_tfp[r] = {}
        energy_utilization[r] = {}
        
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_total_cap = total_capital * reg_weight
        
        sector_weights = {s: 1.0 for s in SECTORS}
        
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
            sector_weights['Transport'] = 4.0
            sector_weights['Agriculture'] = 2.0
            sector_weights['Tourism'] = 2.0
            sector_weights['Metallurgy'] = 0.3
            
        sum_sw = sum(sector_weights.values())
        
        for s in SECTORS:
            share = sector_weights[s] / sum_sw
            initial_capital[r][s] = reg_total_cap * share
            
            base_tfp = 1.0
            if r in ['Kyiv_City']:
                base_tfp = 1.4
            elif r in ['Lviv', 'Dnipro']:
                base_tfp = 1.1
            elif r in ['Donetsk', 'Luhansk', 'Kherson', 'Crimea', 'Sevastopol']:
                base_tfp = 0.4
                
            sector_tfp_mult = {
                'Agriculture': 0.18, 'ConsumerGoods': 0.14, 'Metallurgy': 0.15, 'Energy': 0.22,
                'IT': 0.45, 'Machinery': 0.16, 'MilitaryIndustrial': 0.20, 'Construction': 0.12,
                'Transport': 0.16, 'Retail': 0.14, 'Finance': 0.25, 'Healthcare': 0.10,
                'PublicAdmin': 0.08, 'Chemicals': 0.15, 'Tourism': 0.12
            }
            
            initial_tfp[r][s] = base_tfp * sector_tfp_mult[s] * 18000.0
            
            if r in ['Donetsk', 'Luhansk', 'Kharkiv', 'Kherson', 'Zaporizhzhia']:
                energy_utilization[r][s] = 0.65
            elif r in ['Kyiv_City', 'Kyiv_Oblast', 'Odesa']:
                energy_utilization[r][s] = 0.80
            else:
                energy_utilization[r][s] = 0.92

    prices = {r: {s: 1.0 for s in SECTORS} for r in REGIONS}
    
    # 4. Target demand
    target_final_demand = {}
    for r in REGIONS:
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_total_demand_uah = 5.2e12 * reg_weight
        
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
    
    sum_bs = sum(budget_shares.values())
    budget_shares = {k: v / sum_bs for k, v in budget_shares.items()}
    
    subsistence_demands = {}
    for r in REGIONS:
        subsistence_demands[r] = {}
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_pop = sum(initial_pop[r]['Male'] + initial_pop[r]['Female'])
        
        reg_cons_uah = 3.0e12 * reg_weight
        
        for s in SECTORS:
            share = budget_shares[s]
            subsistence_demands[r][s] = (0.40 * reg_cons_uah * share) / max(1e-1, reg_pop)
            
    wages_by_type = {}
    for r in REGIONS:
        mult = 1.5 if r == 'Kyiv_City' else (1.1 if r in ['Lviv', 'Dnipro'] else 0.8)
        wages_by_type[r] = {
            'unskilled': 120000.0 * mult,
            'semi-skilled': 210000.0 * mult,
            'skilled': 300000.0 * mult
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
        'wages_by_type': wages_by_type,
        'distances': calculate_geographic_distances()
    }
