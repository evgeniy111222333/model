import os
import csv

# Define the 15 parent sectors
PARENT_SECTORS = [
    'Agriculture', 'ConsumerGoods', 'Metallurgy', 'Energy', 'IT',
    'Machinery', 'MilitaryIndustrial', 'Construction', 'Transport',
    'Retail', 'Finance', 'Healthcare', 'PublicAdmin', 'Chemicals', 'Tourism'
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

# Subsector supply weights (how much each subsector supplies within its parent group)
SUPPLY_WEIGHTS = {
    # Agriculture
    'AgriGrain': 0.40, 'AgriTechnical': 0.30, 'AgriLivestock': 0.20, 'Forestry': 0.08, 'Fishery': 0.02,
    
    # Metallurgy
    'SteelIron': 0.45, 'MetalProducts': 0.30, 'NonFerrousMetal': 0.15, 'IronOreMining': 0.10,
    
    # Chemicals (includes Pharma & Biotech)
    'ChemicalFertilizers': 0.12, 'IndustrialChemicals': 0.10, 'PetrochemicalsPlastics': 0.15,
    'BuildingMaterials': 0.13, 'PulpPaper': 0.05, 'NonMetalMining': 0.05,
    'PharmaAPI': 0.05, 'PharmaGenerics': 0.15, 'PharmaOriginals': 0.08,
    'MedicalDevices': 0.06, 'Biotechnologies': 0.04, 'VeterinaryDrugs': 0.02,
    
    # Machinery
    'HeavyMachinery': 0.25, 'TransportMachinery': 0.20, 'AgriMachinery': 0.15,
    'ElectricalEquipment': 0.15, 'PrecisionInstruments': 0.10, 'ElectronicsComponents': 0.10,
    'IndustrialRobots': 0.05,
    
    # MilitaryIndustrial
    'MilSmallArms': 0.15, 'MilArmoredVehicles': 0.25, 'MilArtillery': 0.20, 'MilMissiles': 0.10,
    'MilUAVs': 0.15, 'MilEW': 0.08, 'MilNaval': 0.03, 'MilProtectiveGear': 0.04,
    
    # Energy (includes mining extraction)
    'CoalMining': 0.08, 'OilGasExtraction': 0.12,
    'EnergyThermal': 0.20, 'EnergyNuclearGen': 0.25, 'EnergyNuclearFuel': 0.05,
    'EnergyNuclearWaste': 0.03, 'EnergySolar': 0.07, 'EnergyWindHydro': 0.05,
    'EnergyTransmission': 0.10, 'GasHeatSupply': 0.05,
    
    # Construction
    'ConstResidential': 0.30, 'ConstCommercial': 0.25, 'ConstInfrastructure': 0.25,
    'ConstReconstruction': 0.15, 'RealEstateOps': 0.05,
    
    # ConsumerGoods (includes Food processing, textiles, furniture)
    'FoodProcessing': 0.50, 'TextilesApparel': 0.20, 'LeatherFootwear': 0.15, 'FurnitureHome': 0.15,
    
    # Transport
    'TransRailCargo': 0.30, 'TransRailPassenger': 0.10, 'TransRoad': 0.30,
    'TransWater': 0.05, 'TransAir': 0.05, 'LogisticsWarehouse': 0.20,
    
    # Retail (Trade)
    'TradeWholesale': 0.40, 'TradeRetail': 0.60,
    
    # Finance
    'BankState': 0.25, 'BankCommercial': 0.35, 'BankRetail': 0.20,
    'Insurance': 0.08, 'NonBankFinance': 0.05, 'SecuritiesMarket': 0.05, 'InternationalFinance': 0.02,
    
    # Healthcare
    'HealthPublic': 0.50, 'HealthPrivate': 0.30, 'HealthRehab': 0.15, 'HealthMental': 0.05,
    
    # PublicAdmin (includes Science, Education, Government)
    'PublicAdmin': 0.30, 'LawEnforcement': 0.15, 'UtilityServices': 0.10, 'MilitaryDefense': 0.20,
    'AcadScience': 0.03, 'AppliedRD': 0.07, 'HigherEducation': 0.08, 'GeneralEduVoc': 0.06, 'EdTech': 0.01,
    
    # Tourism (includes food services, hotels, beverages)
    'Beverages': 0.25, 'Tobacco': 0.15, 'FoodServices': 0.40, 'HotelsTourism': 0.20
}

# Baseline 15x15 parent direct requirements coefficients
PARENT_COEFFS = {
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

def generate_and_save_io():
    subsector_names = sorted(list(SUB_SECTORS.keys()))
    
    # 1. Initialize matrix
    matrix = {s2: {s1: 0.0 for s1 in subsector_names} for s2 in subsector_names}
    
    # 2. Populate based on parent flows
    for s2 in subsector_names:
        parent2 = SUB_SECTORS[s2]
        parent_reqs = PARENT_COEFFS.get(parent2, {})
        
        for parent1, coeff in parent_reqs.items():
            # Find all subsectors belonging to parent1
            s1_children = [s for s, p in SUB_SECTORS.items() if p == parent1]
            
            # Normalize supply weights for s1_children to sum to 1
            tot_w = sum(SUPPLY_WEIGHTS.get(s, 1.0) for s in s1_children)
            for s1 in s1_children:
                w_s1 = SUPPLY_WEIGHTS.get(s1, 1.0) / (tot_w if tot_w > 0 else 1.0)
                # s2 consumes s1
                matrix[s2][s1] += coeff * w_s1
                
    # 3. Add custom micro-couplings for high-fidelity realism
    # chemical fertilizers heavily used by crop agriculture
    matrix['AgriGrain']['ChemicalFertilizers'] += 0.08
    matrix['AgriTechnical']['ChemicalFertilizers'] += 0.10
    
    # agricultural machinery used by crop agriculture
    matrix['AgriGrain']['AgriMachinery'] += 0.05
    matrix['AgriTechnical']['AgriMachinery'] += 0.04
    
    # nuclear fuel used by nuclear generation
    matrix['EnergyNuclearGen']['EnergyNuclearFuel'] += 0.15
    matrix['EnergyNuclearFuel']['EnergyNuclearWaste'] += 0.05
    
    # electronic components used by military UAVs, EW, missiles, and precision instruments
    matrix['MilUAVs']['ElectronicsComponents'] += 0.20
    matrix['MilEW']['ElectronicsComponents'] += 0.25
    matrix['MilMissiles']['ElectronicsComponents'] += 0.18
    matrix['PrecisionInstruments']['ElectronicsComponents'] += 0.15
    matrix['IndustrialRobots']['ElectronicsComponents'] += 0.15
    
    # IT services and cybersecurity used by defense systems and financial banks
    matrix['MilEW']['Cybersecurity'] += 0.06
    matrix['MilEW']['ITProductSaaS'] += 0.04
    matrix['BankState']['Cybersecurity'] += 0.08
    matrix['BankCommercial']['Cybersecurity'] += 0.08
    matrix['BankRetail']['Cybersecurity'] += 0.05
    
    # steel/iron used by heavy machinery, transport machinery, and armored vehicles
    matrix['HeavyMachinery']['SteelIron'] += 0.22
    matrix['TransportMachinery']['SteelIron'] += 0.18
    matrix['MilArmoredVehicles']['SteelIron'] += 0.25
    
    # pharma API used by generics/originals
    matrix['PharmaGenerics']['PharmaAPI'] += 0.30
    matrix['PharmaOriginals']['PharmaAPI'] += 0.20
    
    # reconstruction construction consumes building materials
    matrix['ConstReconstruction']['BuildingMaterials'] += 0.28
    matrix['ConstResidential']['BuildingMaterials'] += 0.20
    matrix['ConstCommercial']['BuildingMaterials'] += 0.18
    matrix['ConstInfrastructure']['BuildingMaterials'] += 0.15
    
    # 4. Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'io_ukraine_2026.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Sector'] + subsector_names
        writer.writerow(header)
        for s1 in subsector_names: # supplying sector (row)
            row = [s1]
            for s2 in subsector_names: # consuming sector (column)
                # How much of s1 (row) is needed per unit of s2 (col) output
                row.append(f"{matrix[s2][s1]:.6f}")
            writer.writerow(row)
            
    print(f"Successfully generated {csv_path} with 93x93 sectors.")

if __name__ == '__main__':
    generate_and_save_io()
