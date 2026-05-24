import numpy as np
import os
import csv

# List of 15 parent sectors
PARENT_SECTORS = [
    'Agriculture', 'ConsumerGoods', 'Metallurgy', 'Energy', 'IT',
    'Machinery', 'MilitaryIndustrial', 'Construction', 'Transport',
    'Retail', 'Finance', 'Healthcare', 'PublicAdmin', 'Chemicals', 'Tourism'
]

# List of 27 regions (oblasts, Autonomous Republic of Crimea, Kyiv, Sevastopol)
REGIONS = [
    'Cherkasy', 'Chernihiv', 'Chernivtsi', 'Dnipro', 'Donetsk', 'Ivano-Frankivsk',
    'Kharkiv', 'Kherson', 'Khmelnytskyi', 'Kyiv_Oblast', 'Kirovohrad', 'Luhansk',
    'Lviv', 'Mykolaiv', 'Odesa', 'Poltava', 'Rivne', 'Sumy', 'Ternopil', 'Vinnytsia',
    'Volyn', 'Zakarpattia', 'Zaporizhzhia', 'Zhytomyr', 'Crimea', 'Kyiv_City', 'Sevastopol'
]

# Define 93 sub-sectors and their parent mapping
SUB_SECTORS = {
    # 🌾 Agriculture & Natural Resources (7)
    'AgriGrain': 'Agriculture',
    'AgriTechnical': 'Agriculture',
    'AgriLivestock': 'Agriculture',
    'Forestry': 'Agriculture',
    'Fishery': 'Agriculture',
    'CoalMining': 'Energy',
    'OilGasExtraction': 'Energy',
    
    # ⚙️ Heavy Industry (10)
    'SteelIron': 'Metallurgy',
    'MetalProducts': 'Metallurgy',
    'NonFerrousMetal': 'Metallurgy',
    'IronOreMining': 'Metallurgy',
    'ChemicalFertilizers': 'Chemicals',
    'IndustrialChemicals': 'Chemicals',
    'PetrochemicalsPlastics': 'Chemicals',
    'BuildingMaterials': 'Chemicals',
    'PulpPaper': 'Chemicals',
    'NonMetalMining': 'Chemicals',
    
    # 🔬 Pharma & Biotechnology (6)
    'PharmaAPI': 'Chemicals',
    'PharmaGenerics': 'Chemicals',
    'PharmaOriginals': 'Chemicals',
    'MedicalDevices': 'Chemicals',
    'Biotechnologies': 'Chemicals',
    'VeterinaryDrugs': 'Chemicals',
    
    # 🏭 Machinery (7)
    'HeavyMachinery': 'Machinery',
    'TransportMachinery': 'Machinery',
    'AgriMachinery': 'Machinery',
    'ElectricalEquipment': 'Machinery',
    'PrecisionInstruments': 'Machinery',
    'ElectronicsComponents': 'Machinery',
    'IndustrialRobots': 'Machinery',
    
    # 🪖 Military-Industrial Complex (8)
    'MilSmallArms': 'MilitaryIndustrial',
    'MilArmoredVehicles': 'MilitaryIndustrial',
    'MilArtillery': 'MilitaryIndustrial',
    'MilMissiles': 'MilitaryIndustrial',
    'MilUAVs': 'MilitaryIndustrial',
    'MilEW': 'MilitaryIndustrial',
    'MilNaval': 'MilitaryIndustrial',
    'MilProtectiveGear': 'MilitaryIndustrial',
    
    # ⚡ Energy (7)
    'EnergyThermal': 'Energy',
    'EnergyNuclearGen': 'Energy',
    'EnergyNuclearFuel': 'Energy',
    'EnergyNuclearWaste': 'Energy',
    'EnergySolar': 'Energy',
    'EnergyWindHydro': 'Energy',
    'EnergyTransmission': 'Energy',
    
    # 🏗️ Construction & Real Estate (5)
    'ConstResidential': 'Construction',
    'ConstCommercial': 'Construction',
    'ConstInfrastructure': 'Construction',
    'ConstReconstruction': 'Construction',
    'RealEstateOps': 'Construction',
    
    # 🍞 Light & Food Industry (6)
    'FoodProcessing': 'ConsumerGoods',
    'Beverages': 'Tourism',
    'Tobacco': 'Tourism',
    'TextilesApparel': 'ConsumerGoods',
    'LeatherFootwear': 'ConsumerGoods',
    'FurnitureHome': 'ConsumerGoods',
    
    # 🚛 Transport & Logistics (6)
    'TransRailCargo': 'Transport',
    'TransRailPassenger': 'Transport',
    'TransRoad': 'Transport',
    'TransWater': 'Transport',
    'TransAir': 'Transport',
    'LogisticsWarehouse': 'Transport',
    
    # 🛒 Trade & Consumer Services (4)
    'TradeWholesale': 'Retail',
    'TradeRetail': 'Retail',
    'FoodServices': 'Tourism',
    'HotelsTourism': 'Tourism',
    
    # 💻 IT & Telecom (6)
    'ITServicesExport': 'IT',
    'ITProductSaaS': 'IT',
    'Telecom': 'IT',
    'InternetCloud': 'IT',
    'MediaAdvertising': 'IT',
    'Cybersecurity': 'IT',
    
    # 🏦 Banking & Finance (7)
    'BankState': 'Finance',
    'BankCommercial': 'Finance',
    'BankRetail': 'Finance',
    'Insurance': 'Finance',
    'NonBankFinance': 'Finance',
    'SecuritiesMarket': 'Finance',
    'InternationalFinance': 'Finance',
    
    # 🔬 Science, R&D & Education (5)
    'AcadScience': 'PublicAdmin',
    'AppliedRD': 'PublicAdmin',
    'HigherEducation': 'PublicAdmin',
    'GeneralEduVoc': 'PublicAdmin',
    'EdTech': 'PublicAdmin',
    
    # 🏥 Healthcare (4)
    'HealthPublic': 'Healthcare',
    'HealthPrivate': 'Healthcare',
    'HealthRehab': 'Healthcare',
    'HealthMental': 'Healthcare',
    
    # 🏛️ Public Sector & Security (5)
    'PublicAdmin': 'PublicAdmin',
    'LawEnforcement': 'PublicAdmin',
    'UtilityServices': 'PublicAdmin',
    'GasHeatSupply': 'PublicAdmin',
    'MilitaryDefense': 'PublicAdmin'
}

SECTORS = sorted(list(SUB_SECTORS.keys()))

# Subsector supply weights (used to distribute parent variables)
SUPPLY_WEIGHTS = {
    'AgriGrain': 0.40, 'AgriTechnical': 0.30, 'AgriLivestock': 0.20, 'Forestry': 0.08, 'Fishery': 0.02,
    'SteelIron': 0.45, 'MetalProducts': 0.30, 'NonFerrousMetal': 0.15, 'IronOreMining': 0.10,
    'ChemicalFertilizers': 0.12, 'IndustrialChemicals': 0.10, 'PetrochemicalsPlastics': 0.15,
    'BuildingMaterials': 0.13, 'PulpPaper': 0.05, 'NonMetalMining': 0.05,
    'PharmaAPI': 0.05, 'PharmaGenerics': 0.15, 'PharmaOriginals': 0.08,
    'MedicalDevices': 0.06, 'Biotechnologies': 0.04, 'VeterinaryDrugs': 0.02,
    'HeavyMachinery': 0.25, 'TransportMachinery': 0.20, 'AgriMachinery': 0.15,
    'ElectricalEquipment': 0.15, 'PrecisionInstruments': 0.10, 'ElectronicsComponents': 0.10,
    'IndustrialRobots': 0.05,
    'MilSmallArms': 0.15, 'MilArmoredVehicles': 0.25, 'MilArtillery': 0.20, 'MilMissiles': 0.10,
    'MilUAVs': 0.15, 'MilEW': 0.08, 'MilNaval': 0.03, 'MilProtectiveGear': 0.04,
    'CoalMining': 0.08, 'OilGasExtraction': 0.12,
    'EnergyThermal': 0.20, 'EnergyNuclearGen': 0.25, 'EnergyNuclearFuel': 0.05,
    'EnergyNuclearWaste': 0.03, 'EnergySolar': 0.07, 'EnergyWindHydro': 0.05,
    'EnergyTransmission': 0.10, 'GasHeatSupply': 0.05,
    'ConstResidential': 0.30, 'ConstCommercial': 0.25, 'ConstInfrastructure': 0.25,
    'ConstReconstruction': 0.15, 'RealEstateOps': 0.05,
    'FoodProcessing': 0.50, 'TextilesApparel': 0.20, 'LeatherFootwear': 0.15, 'FurnitureHome': 0.15,
    'TransRailCargo': 0.30, 'TransRailPassenger': 0.10, 'TransRoad': 0.30,
    'TransWater': 0.05, 'TransAir': 0.05, 'LogisticsWarehouse': 0.20,
    'TradeWholesale': 0.40, 'TradeRetail': 0.60,
    'BankState': 0.25, 'BankCommercial': 0.35, 'BankRetail': 0.20,
    'Insurance': 0.08, 'NonBankFinance': 0.05, 'SecuritiesMarket': 0.05, 'InternationalFinance': 0.02,
    'HealthPublic': 0.50, 'HealthPrivate': 0.30, 'HealthRehab': 0.15, 'HealthMental': 0.05,
    'PublicAdmin': 0.30, 'LawEnforcement': 0.15, 'UtilityServices': 0.10, 'MilitaryDefense': 0.20,
    'AcadScience': 0.03, 'AppliedRD': 0.07, 'HigherEducation': 0.08, 'GeneralEduVoc': 0.06, 'EdTech': 0.01,
    'Beverages': 0.25, 'Tobacco': 0.15, 'FoodServices': 0.40, 'HotelsTourism': 0.20
}

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
    'Kyiv_Oblast': (50.4501, 30.5234),
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
    Loads direct requirements technical coefficients matrix for the 93 sectors
    from data/io_ukraine_2026.csv.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'io_ukraine_2026.csv')
    if os.path.exists(csv_path):
        coefficients = {s: {} for s in SECTORS}
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                supplying_sector = row['Sector']
                if supplying_sector not in SECTORS:
                    continue
                for consuming_sector in SECTORS:
                    val = float(row.get(consuming_sector, 0.0))
                    if val > 0:
                        coefficients[consuming_sector][supplying_sector] = val
        return coefficients
    else:
        # Emergency fallback (should not occur since we generate the file)
        return {s: {} for s in SECTORS}

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
                dist[i, j] = max(1.0, R * c * 1.3) # 30% curvature adjustment, floored at 1.0 km to prevent division by zero
    return dist

def generate_baseline_data():
    """
    Procedurally synthesizes the baseline state for Ukraine's economy in 2026.
    Uses 18 five-year cohorts, real coordinates, and 3 labor types for 93 sectors.
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
    
    cohort_shares_male = np.array([
        0.052, 0.051, 0.050, 0.055, 0.062, 0.068, 0.072, 0.075, 0.072, 
        0.068, 0.062, 0.058, 0.052, 0.048, 0.040, 0.030, 0.020, 0.015
    ])
    cohort_shares_female = np.array([
        0.048, 0.047, 0.046, 0.050, 0.058, 0.064, 0.068, 0.072, 0.070, 
        0.068, 0.065, 0.062, 0.058, 0.055, 0.050, 0.042, 0.032, 0.025
    ])
    
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
        
        if r in ['Kyiv_City', 'Kharkiv', 'Odesa', 'Lviv']:
            fertility_rates[r] = 0.85
        elif r in ['Zakarpattia', 'Volyn', 'Rivne', 'Ivano-Frankivsk']:
            fertility_rates[r] = 1.15
        else:
            fertility_rates[r] = 1.0
            
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
        
        # Parent weights
        parent_weights = {p: 1.0 for p in PARENT_SECTORS}
        
        if r == 'Kyiv_City':
            parent_weights['IT'] = 5.0
            parent_weights['Finance'] = 4.0
            parent_weights['Retail'] = 3.0
            parent_weights['PublicAdmin'] = 3.0
            parent_weights['Agriculture'] = 0.05
            parent_weights['Metallurgy'] = 0.05
        elif r in ['Dnipro', 'Zaporizhzhia', 'Donetsk', 'Luhansk']:
            parent_weights['Metallurgy'] = 5.0
            parent_weights['Machinery'] = 3.0
            parent_weights['Energy'] = 3.0
            parent_weights['IT'] = 0.5
        elif r in ['Poltava', 'Vinnytsia', 'Cherkasy', 'Kirovohrad', 'Khmelnytskyi']:
            parent_weights['Agriculture'] = 4.0
            parent_weights['ConsumerGoods'] = 2.0
            parent_weights['Metallurgy'] = 0.1
        elif r in ['Lviv', 'Kharkiv']:
            parent_weights['IT'] = 3.0
            parent_weights['Machinery'] = 2.0
            parent_weights['Tourism'] = 2.5
            parent_weights['Agriculture'] = 0.5
        elif r in ['Odesa', 'Mykolaiv', 'Kherson']:
            parent_weights['Transport'] = 4.0
            parent_weights['Agriculture'] = 2.0
            parent_weights['Tourism'] = 2.0
            parent_weights['Metallurgy'] = 0.3
            
        # Distribute parent weights to sub-sectors
        subsector_weights = {}
        for s in SECTORS:
            p = SUB_SECTORS[s]
            p_w = parent_weights.get(p, 1.0)
            sub_w = SUPPLY_WEIGHTS.get(s, 1.0)
            
            # Normalize subsector weights per parent group
            siblings = [sib for sib, parent in SUB_SECTORS.items() if parent == p]
            tot_sib_w = sum(SUPPLY_WEIGHTS.get(sib, 1.0) for sib in siblings)
            norm_sub_w = sub_w / (tot_sib_w if tot_sib_w > 0 else 1.0)
            
            subsector_weights[s] = p_w * norm_sub_w
            
        sum_sw = sum(subsector_weights.values())
        
        for s in SECTORS:
            share = subsector_weights[s] / sum_sw
            initial_capital[r][s] = reg_total_cap * share
            
            base_tfp = 1.0
            if r in ['Kyiv_City']:
                base_tfp = 1.4
            elif r in ['Lviv', 'Dnipro']:
                base_tfp = 1.1
            elif r in ['Donetsk', 'Luhansk', 'Kherson', 'Crimea', 'Sevastopol']:
                base_tfp = 0.4
                
            parent = SUB_SECTORS[s]
            parent_tfp_mult = {
                'Agriculture': 0.18, 'ConsumerGoods': 0.14, 'Metallurgy': 0.15, 'Energy': 0.22,
                'IT': 0.45, 'Machinery': 0.16, 'MilitaryIndustrial': 0.20, 'Construction': 0.12,
                'Transport': 0.16, 'Retail': 0.14, 'Finance': 0.25, 'Healthcare': 0.10,
                'PublicAdmin': 0.08, 'Chemicals': 0.15, 'Tourism': 0.12
            }.get(parent, 0.15)
            
            initial_tfp[r][s] = base_tfp * parent_tfp_mult * 18000.0
            
            if r in ['Donetsk', 'Luhansk', 'Kharkiv', 'Kherson', 'Zaporizhzhia']:
                energy_utilization[r][s] = 0.65
            elif r in ['Kyiv_City', 'Kyiv_Oblast', 'Odesa']:
                energy_utilization[r][s] = 0.80
            else:
                energy_utilization[r][s] = 0.92

    prices = {r: {s: 1.0 for s in SECTORS} for r in REGIONS}
    
    # 4. Target demand
    target_final_demand = {}
    parent_demand_weights = {
        'Agriculture': 0.08, 'ConsumerGoods': 0.25, 'Metallurgy': 0.05, 'Energy': 0.06,
        'IT': 0.08, 'Machinery': 0.05, 'MilitaryIndustrial': 0.08, 'Construction': 0.08,
        'Transport': 0.06, 'Retail': 0.05, 'Finance': 0.04, 'Healthcare': 0.05,
        'PublicAdmin': 0.04, 'Chemicals': 0.02, 'Tourism': 0.01
    }
    
    for r in REGIONS:
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_total_demand_uah = 5.2e12 * reg_weight
        
        # Distribute parent demand weights to sub-sectors
        subsector_demand_weights = {}
        for s in SECTORS:
            p = SUB_SECTORS[s]
            p_dw = parent_demand_weights.get(p, 0.05)
            sub_w = SUPPLY_WEIGHTS.get(s, 1.0)
            siblings = [sib for sib, parent in SUB_SECTORS.items() if parent == p]
            tot_sib_w = sum(SUPPLY_WEIGHTS.get(sib, 1.0) for sib in siblings)
            norm_sub_w = sub_w / (tot_sib_w if tot_sib_w > 0 else 1.0)
            
            subsector_demand_weights[s] = p_dw * norm_sub_w
            
        sum_sdw = sum(subsector_demand_weights.values())
        
        for s in SECTORS:
            share = subsector_demand_weights[s] / sum_sdw
            target_final_demand[(r, s)] = reg_total_demand_uah * share

    # 5. ABM / CGE Parameters
    # Calculate base shares
    base_shares = {}
    for s in SECTORS:
        p = SUB_SECTORS[s]
        p_dw = parent_demand_weights.get(p, 0.05)
        sub_w = SUPPLY_WEIGHTS.get(s, 1.0)
        siblings = [sib for sib, parent in SUB_SECTORS.items() if parent == p]
        tot_sib_w = sum(SUPPLY_WEIGHTS.get(sib, 1.0) for sib in siblings)
        norm_sub_w = sub_w / (tot_sib_w if tot_sib_w > 0 else 1.0)
        base_shares[s] = p_dw * norm_sub_w
    sum_bs = sum(base_shares.values())
    base_shares = {k: v / sum_bs for k, v in base_shares.items()}
    
    budget_shares = {}
    for r in REGIONS:
        budget_shares[r] = {}
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        
        r_budget_shares = {}
        for s in SECTORS:
            p = SUB_SECTORS[s]
            base_share = base_shares[s]
            
            is_essential = p in ['Agriculture', 'Healthcare', 'Energy'] or s in ['FoodProcessing', 'UtilityServices', 'GasHeatSupply']
            if is_essential:
                r_budget_shares[s] = base_share * (1.2 - 0.15 * min(4.0, reg_weight * len(REGIONS)))
            else:
                r_budget_shares[s] = base_share * (0.8 + 0.15 * min(4.0, reg_weight * len(REGIONS)))
            r_budget_shares[s] = max(1e-5, r_budget_shares[s])
            
        sum_rbs = sum(r_budget_shares.values())
        budget_shares[r] = {k: v / sum_rbs for k, v in r_budget_shares.items()}
    
    subsistence_demands = {}
    for r in REGIONS:
        subsistence_demands[r] = {}
        reg_weight = reg_grp_weights.get(r, 0.01) / total_grp_weight
        reg_pop = sum(initial_pop[r]['Male'] + initial_pop[r]['Female'])
        
        reg_cons_uah = 3.0e12 * reg_weight
        
        for s in SECTORS:
            share = budget_shares[r][s]
            parent = SUB_SECTORS[s]
            if parent in ['Agriculture', 'Healthcare'] or s in [
                'FoodProcessing', 'UtilityServices', 'GasHeatSupply', 'EnergyThermal',
                'EnergyNuclearGen', 'EnergyTransmission', 'Telecom', 'PharmaGenerics',
                'PharmaOriginals', 'TransRoad', 'TransRailPassenger', 'ConstResidential', 'TradeRetail'
            ]:
                sub_coeff = 0.75
            else:
                sub_coeff = 0.05
            subsistence_demands[r][s] = (sub_coeff * reg_cons_uah * share) / max(1e-1, reg_pop)
            
    wages_by_type = {}
    for r in REGIONS:
        # Regional wage multipliers based on economic development level
        # Kyiv: 1.5x (capital, finance/IT hub)
        # Regional centers: 1.1-1.2x (Lviv, Dnipro, Odesa, Kharkiv)
        # Mid-tier: 0.95x (larger oblast centers)
        # Rural/peripheral: 0.75-0.85x
        if r == 'Kyiv_City':
            mult = 1.5
        elif r in ['Lviv', 'Dnipro', 'Odesa', 'Kharkiv', 'Kyiv_Oblast']:
            mult = 1.1
        elif r in ['Poltava', 'Vinnytsia', 'Cherkasy', 'Zaporizhzhia', 'Mykolaiv']:
            mult = 0.95
        elif r in ['Sumy', 'Chernihiv', 'Kirovohrad', 'Khmelnytskyi', 'Zhytomyr']:
            mult = 0.85
        else:
            mult = 0.75  # Rural/peripheral regions
        
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
