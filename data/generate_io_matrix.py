import os
import csv
import numpy as np

# List of 27 regions (for reference)
REGIONS = [
    'Cherkasy', 'Chernihiv', 'Chernivtsi', 'Dnipro', 'Donetsk', 'Ivano-Frankivsk',
    'Kharkiv', 'Kherson', 'Khmelnytskyi', 'Kyiv_Oblast', 'Kirovohrad', 'Luhansk',
    'Lviv', 'Mykolaiv', 'Odesa', 'Poltava', 'Rivne', 'Sumy', 'Ternopil', 'Vinnytsia',
    'Volyn', 'Zakarpattia', 'Zaporizhzhia', 'Zhytomyr', 'Crimea', 'Kyiv_City', 'Sevastopol'
]

# Define 93 sub-sectors
SUB_SECTORS = [
    # 🌾 Agriculture & Natural Resources
    'AgriGrain', 'AgriTechnical', 'AgriLivestock', 'Forestry', 'Fishery', 'CoalMining', 'OilGasExtraction',
    # ⚙️ Heavy Industry
    'SteelIron', 'MetalProducts', 'NonFerrousMetal', 'IronOreMining', 'ChemicalFertilizers', 
    'IndustrialChemicals', 'PetrochemicalsPlastics', 'BuildingMaterials', 'PulpPaper', 'NonMetalMining',
    # 🔬 Pharma & Biotechnology
    'PharmaAPI', 'PharmaGenerics', 'PharmaOriginals', 'MedicalDevices', 'Biotechnologies', 'VeterinaryDrugs',
    # 🏭 Machinery
    'HeavyMachinery', 'TransportMachinery', 'AgriMachinery', 'ElectricalEquipment', 'PrecisionInstruments', 
    'ElectronicsComponents', 'IndustrialRobots',
    # 🪖 Military-Industrial Complex
    'MilSmallArms', 'MilArmoredVehicles', 'MilArtillery', 'MilMissiles', 'MilUAVs', 'MilEW', 'MilNaval', 'MilProtectiveGear',
    # ⚡ Energy
    'EnergyThermal', 'EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste', 'EnergySolar', 'EnergyWindHydro', 'EnergyTransmission',
    # 🏗️ Construction & Real Estate
    'ConstResidential', 'ConstCommercial', 'ConstInfrastructure', 'ConstReconstruction', 'RealEstateOps',
    # 🍞 Light & Food Industry
    'FoodProcessing', 'Beverages', 'Tobacco', 'TextilesApparel', 'LeatherFootwear', 'FurnitureHome',
    # 🚛 Transport & Logistics
    'TransRailCargo', 'TransRailPassenger', 'TransRoad', 'TransWater', 'TransAir', 'LogisticsWarehouse',
    # 🛒 Trade & Consumer Services
    'TradeWholesale', 'TradeRetail', 'FoodServices', 'HotelsTourism',
    # 💻 IT & Telecom
    'ITServicesExport', 'ITProductSaaS', 'Telecom', 'InternetCloud', 'MediaAdvertising', 'Cybersecurity',
    # 🏦 Banking & Finance
    'BankState', 'BankCommercial', 'BankRetail', 'Insurance', 'NonBankFinance', 'SecuritiesMarket', 'InternationalFinance',
    # 🔬 Science, R&D & Education
    'AcadScience', 'AppliedRD', 'HigherEducation', 'GeneralEduVoc', 'EdTech',
    # 🏥 Healthcare
    'HealthPublic', 'HealthPrivate', 'HealthRehab', 'HealthMental',
    # 🏛️ Public Sector & Security
    'PublicAdmin', 'LawEnforcement', 'UtilityServices', 'GasHeatSupply', 'MilitaryDefense'
]

# Set seed for reproducibility
np.random.seed(12345)

def generate_io_matrix():
    n = len(SUB_SECTORS)
    matrix = np.zeros((n, n))
    
    # 1. Fill background with sparse unique technical noise
    # Represents realistic sparsity (most sectors only consume from a subset of other sectors)
    # Spreads values across a wider range to prevent collision at 6 decimal places
    for j in range(n):
        matrix[j, j] = np.random.uniform(0.015000, 0.035000) # self-consumption
        
        # Choose a random 25% of other rows to be non-zero background inputs
        other_indices = [i for i in range(n) if i != j]
        num_aux = int(n * 0.25)
        aux_indices = np.random.choice(other_indices, size=num_aux, replace=False)
        
        for i in aux_indices:
            matrix[i, j] = np.random.uniform(0.000800, 0.009000)
            
    # 2. Inject Group-Level and Sector-Specific Key Production Functions
    for j, col_sector in enumerate(SUB_SECTORS):
        key_inputs = {}
        target_gva = 0.45 # default GVA target (45%), meaning sum of inputs = 0.55
        
        # 🌾 Crop Agriculture
        if col_sector in ['AgriGrain', 'AgriTechnical']:
            key_inputs['ChemicalFertilizers'] = 0.14
            key_inputs['AgriMachinery'] = 0.08
            key_inputs['TransRoad'] = 0.05
            key_inputs['TradeWholesale'] = 0.04
            key_inputs['EnergyThermal'] = 0.03
            key_inputs['IndustrialChemicals'] = 0.02
            target_gva = 0.50
            
        # 🌾 Livestock
        elif col_sector == 'AgriLivestock':
            key_inputs['AgriGrain'] = 0.16
            key_inputs['VeterinaryDrugs'] = 0.06
            key_inputs['FoodProcessing'] = 0.04
            key_inputs['TransRoad'] = 0.03
            target_gva = 0.45
            
        # ⛏️ Mining Extraction
        elif col_sector in ['CoalMining', 'OilGasExtraction', 'IronOreMining', 'NonMetalMining']:
            key_inputs['HeavyMachinery'] = 0.12
            key_inputs['ElectricalEquipment'] = 0.07
            key_inputs['EnergyTransmission'] = 0.06
            key_inputs['TransRailCargo'] = 0.05
            key_inputs['IndustrialChemicals'] = 0.03 # explosives
            target_gva = 0.55
            
        # ⚙️ Metallurgy
        elif col_sector == 'SteelIron':
            key_inputs['IronOreMining'] = 0.18
            key_inputs['CoalMining'] = 0.15
            key_inputs['EnergyThermal'] = 0.10
            key_inputs['TransRailCargo'] = 0.05
            target_gva = 0.38
        elif col_sector in ['MetalProducts', 'NonFerrousMetal']:
            key_inputs['SteelIron'] = 0.22
            key_inputs['EnergyThermal'] = 0.08
            key_inputs['IndustrialChemicals'] = 0.04
            target_gva = 0.40
            
        # 🔬 Pharma & Biotech
        elif col_sector in ['PharmaAPI', 'PharmaGenerics', 'PharmaOriginals']:
            key_inputs['IndustrialChemicals'] = 0.15
            key_inputs['PetrochemicalsPlastics'] = 0.08
            key_inputs['ITProductSaaS'] = 0.05
            key_inputs['AppliedRD'] = 0.06
            target_gva = 0.52 # High GVA (high margins/salaries)
            
        # 🔬 Biotechnologies
        elif col_sector == 'Biotechnologies':
            key_inputs['Biotechnologies'] = 0.08
            key_inputs['AcadScience'] = 0.10
            key_inputs['PharmaAPI'] = 0.05
            target_gva = 0.60 # Very High GVA
            
        # 🏭 Heavy Machinery / Transport Machinery
        elif col_sector in ['HeavyMachinery', 'TransportMachinery']:
            key_inputs['SteelIron'] = 0.16
            key_inputs['MetalProducts'] = 0.08
            key_inputs['ElectricalEquipment'] = 0.06
            key_inputs['PrecisionInstruments'] = 0.04
            key_inputs['IndustrialChemicals'] = 0.02
            target_gva = 0.40
            
        # 🚜 Agricultural Machinery
        elif col_sector == 'AgriMachinery':
            key_inputs['SteelIron'] = 0.14
            key_inputs['MetalProducts'] = 0.06
            key_inputs['ElectronicsComponents'] = 0.05
            key_inputs['ElectricalEquipment'] = 0.04
            target_gva = 0.45
            
        # 🔌 Electrical & Electronics
        elif col_sector in ['ElectricalEquipment', 'ElectronicsComponents', 'PrecisionInstruments', 'IndustrialRobots']:
            key_inputs['NonFerrousMetal'] = 0.12 # copper, aluminum
            key_inputs['MetalProducts'] = 0.06
            key_inputs['ElectronicsComponents'] = 0.10
            key_inputs['ITProductSaaS'] = 0.05
            target_gva = 0.48
            
        # 🪖 VPC - Small Arms / Protection
        elif col_sector == 'MilSmallArms':
            key_inputs['SteelIron'] = 0.15
            key_inputs['IndustrialChemicals'] = 0.08 # propellants
            key_inputs['MetalProducts'] = 0.06
            key_inputs['LogisticsWarehouse'] = 0.03
            target_gva = 0.45
        elif col_sector == 'MilProtectiveGear':
            key_inputs['TextilesApparel'] = 0.18 # Kevlar
            key_inputs['NonFerrousMetal'] = 0.08 # armor plates
            key_inputs['PetrochemicalsPlastics'] = 0.08
            target_gva = 0.50
            
        # 🪖 VPC - Armored Vehicles
        elif col_sector == 'MilArmoredVehicles':
            key_inputs['SteelIron'] = 0.16
            key_inputs['MetalProducts'] = 0.08
            key_inputs['HeavyMachinery'] = 0.10
            key_inputs['TransportMachinery'] = 0.05
            key_inputs['ElectronicsComponents'] = 0.06
            key_inputs['MilProtectiveGear'] = 0.04
            target_gva = 0.38
            
        # 🪖 VPC - Artillery
        elif col_sector == 'MilArtillery':
            key_inputs['SteelIron'] = 0.22
            key_inputs['MetalProducts'] = 0.08
            key_inputs['HeavyMachinery'] = 0.10
            key_inputs['IndustrialChemicals'] = 0.06
            key_inputs['LogisticsWarehouse'] = 0.02
            target_gva = 0.40
            
        # 🪖 VPC - Missiles & Rockets
        elif col_sector == 'MilMissiles':
            key_inputs['NonFerrousMetal'] = 0.12 # aerospace titanium/aluminum
            key_inputs['ElectronicsComponents'] = 0.10
            key_inputs['MilEW'] = 0.06
            key_inputs['IndustrialChemicals'] = 0.08 # solid fuel
            key_inputs['PrecisionInstruments'] = 0.05
            target_gva = 0.45
            
        # 🪖 VPC - UAVs (Drones)
        elif col_sector == 'MilUAVs':
            key_inputs['ElectronicsComponents'] = 0.18 # microchips, flight controllers
            key_inputs['PetrochemicalsPlastics'] = 0.08 # carbon fiber, fuselage
            key_inputs['ITProductSaaS'] = 0.08 # flight software
            key_inputs['ElectricalEquipment'] = 0.04
            key_inputs['LogisticsWarehouse'] = 0.03
            target_gva = 0.50 # High wage share / value added
            
        # 🪖 VPC - EW (Radio-Electronic Warfare)
        elif col_sector == 'MilEW':
            key_inputs['ElectronicsComponents'] = 0.20 # RF components
            key_inputs['Cybersecurity'] = 0.08 # encryption
            key_inputs['ITProductSaaS'] = 0.06 # analysis software
            key_inputs['PrecisionInstruments'] = 0.06
            key_inputs['ElectricalEquipment'] = 0.05
            target_gva = 0.50 # 50% GVA, leaving 50% for intermediate inputs
            
        # 🪖 VPC - Naval defense
        elif col_sector == 'MilNaval':
            key_inputs['SteelIron'] = 0.15
            key_inputs['TransportMachinery'] = 0.18 # ship hulls
            key_inputs['ElectronicsComponents'] = 0.05
            key_inputs['MilEW'] = 0.05
            target_gva = 0.42
            
        # ⚡ Energy - Thermal
        elif col_sector == 'EnergyThermal':
            key_inputs['CoalMining'] = 0.22
            key_inputs['OilGasExtraction'] = 0.10
            key_inputs['EnergyTransmission'] = 0.04
            target_gva = 0.45
            
        # ⚡ Energy - Nuclear Generation
        elif col_sector == 'EnergyNuclearGen':
            key_inputs['EnergyNuclearFuel'] = 0.18
            key_inputs['EnergyNuclearWaste'] = 0.06
            key_inputs['EnergyTransmission'] = 0.05
            target_gva = 0.55
            
        # ⚡ Energy - Renewables (Solar & Wind/Hydro)
        elif col_sector in ['EnergySolar', 'EnergyWindHydro']:
            key_inputs['ElectricalEquipment'] = 0.12
            key_inputs['ElectronicsComponents'] = 0.06
            key_inputs['BuildingMaterials'] = 0.05
            target_gva = 0.65 # Very high profit/depreciation margins
            
        # ⚡ Energy - Grid & Transmission
        elif col_sector == 'EnergyTransmission':
            key_inputs['ElectricalEquipment'] = 0.15 # lines, transformers
            key_inputs['EnergyTransmission'] = 0.06 # losses
            target_gva = 0.50
            
        # 🏗️ Construction & Infrastructure
        elif col_sector in ['ConstResidential', 'ConstCommercial', 'ConstInfrastructure', 'ConstReconstruction']:
            key_inputs['BuildingMaterials'] = 0.18
            key_inputs['SteelIron'] = 0.08
            key_inputs['MetalProducts'] = 0.05
            key_inputs['TransRoad'] = 0.04
            key_inputs['TradeWholesale'] = 0.03
            target_gva = 0.48
            
        # 🍞 Food Processing & Light Industry
        elif col_sector == 'FoodProcessing':
            key_inputs['AgriLivestock'] = 0.15
            key_inputs['AgriGrain'] = 0.12
            key_inputs['PetrochemicalsPlastics'] = 0.04 # packaging
            key_inputs['TransRoad'] = 0.03
            target_gva = 0.40
        elif col_sector == 'TextilesApparel':
            key_inputs['PetrochemicalsPlastics'] = 0.12 # synthetic fiber
            key_inputs['TransRoad'] = 0.04
            target_gva = 0.42
            
        # 🚛 Transport & Logistics
        elif col_sector in ['TransRailCargo', 'TransRailPassenger', 'TransRoad', 'TransWater', 'TransAir']:
            key_inputs['OilGasExtraction'] = 0.15 # fuel
            key_inputs['TransportMachinery'] = 0.06 # maintenance
            key_inputs['LogisticsWarehouse'] = 0.03
            target_gva = 0.52
        elif col_sector == 'LogisticsWarehouse':
            key_inputs['TransRoad'] = 0.08
            key_inputs['InternetCloud'] = 0.04
            target_gva = 0.55
            
        # 🛒 Trade & Retail
        elif col_sector in ['TradeWholesale', 'TradeRetail']:
            key_inputs['TransRoad'] = 0.05
            key_inputs['RealEstateOps'] = 0.04
            key_inputs['LogisticsWarehouse'] = 0.03
            target_gva = 0.62 # high retail margins
            
        # 💻 IT & Software (Export vs SaaS)
        elif col_sector in ['ITServicesExport', 'ITProductSaaS']:
            key_inputs['InternetCloud'] = 0.08
            key_inputs['Telecom'] = 0.04
            key_inputs['ElectronicsComponents'] = 0.03
            target_gva = 0.72 # Extremely high GVA (pure labor/wages)
        elif col_sector == 'Cybersecurity':
            key_inputs['InternetCloud'] = 0.06
            key_inputs['ElectronicsComponents'] = 0.04
            target_gva = 0.68
            
        # 🏦 Banking & Finance
        elif col_sector in ['BankState', 'BankCommercial', 'BankRetail']:
            key_inputs['Cybersecurity'] = 0.05
            key_inputs['ITProductSaaS'] = 0.04
            key_inputs['Telecom'] = 0.03
            key_inputs['RealEstateOps'] = 0.03
            target_gva = 0.65
            
        # 🏥 Healthcare (Public / Private / Rehab)
        elif col_sector in ['HealthPublic', 'HealthPrivate', 'HealthRehab', 'HealthMental']:
            key_inputs['PharmaGenerics'] = 0.08
            key_inputs['MedicalDevices'] = 0.05
            key_inputs['UtilityServices'] = 0.03
            target_gva = 0.68 # Heavy labor-oriented
            
        # 🏛️ Public Admin & Defense
        elif col_sector == 'MilitaryDefense':
            key_inputs['MilSmallArms'] = 0.04
            key_inputs['MilArmoredVehicles'] = 0.04
            key_inputs['MilArtillery'] = 0.03
            key_inputs['MilUAVs'] = 0.03
            key_inputs['MilEW'] = 0.02
            key_inputs['MilProtectiveGear'] = 0.02
            key_inputs['LogisticsWarehouse'] = 0.03
            target_gva = 0.70 # Mostly wages of personnel
            
        # Fill matrix for key inputs
        for input_name, base_coeff in key_inputs.items():
            if input_name in SUB_SECTORS:
                row_idx = SUB_SECTORS.index(input_name)
                # Perturb the base coefficient by +/- 10% to ensure uniqueness
                matrix[row_idx, j] = base_coeff * np.random.uniform(0.90, 1.10)
                
        # 3. Column Normalization
        # Sum of intermediate inputs = 1.0 - target_gva
        # We scale the column (except the diagonal self-consumption to preserve structural properties)
        target_sum = 1.0 - target_gva
        
        # Current sum of column
        col_sum = np.sum(matrix[:, j])
        
        # Scaling factor
        if col_sum > 0:
            scale = target_sum / col_sum
            matrix[:, j] *= scale
            
        # Ensure self-consumption stays within reasonable bounds [0.015, 0.05]
        matrix[j, j] = np.clip(matrix[j, j], 0.015, 0.05)
        
    # 4. Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'io_ukraine_2026.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Sector'] + SUB_SECTORS
        writer.writerow(header)
        for i, s1 in enumerate(SUB_SECTORS): # supplying (row)
            row = [s1]
            for j, s2 in enumerate(SUB_SECTORS): # consuming (col)
                row.append(f"{matrix[i, j]:.7f}")
            writer.writerow(row)
            
    # Verify properties
    print(f"Matrix shape: {matrix.shape}")
    
    # Check uniqueness
    flat_matrix = matrix.flatten()
    unique_vals = len(np.unique(flat_matrix))
    total_vals = len(flat_matrix)
    print(f"Total cells: {total_vals}")
    print(f"Unique values: {unique_vals}")
    print(f"Uniqueness ratio: {unique_vals / total_vals:.4f} (Target: > 0.85)")
    
    # Check sums for MilEW, MilUAVs, MilArtillery
    for name in ['MilEW', 'MilUAVs', 'MilArtillery']:
        idx = SUB_SECTORS.index(name)
        col_sum = np.sum(matrix[:, idx])
        print(f"Sector {name}: Sum of intermediate inputs = {col_sum:.4f} (GVA = {1.0 - col_sum:.4f})")
        # Print top inputs
        top_indices = np.argsort(matrix[:, idx])[::-1][:4]
        top_inputs = [f"{SUB_SECTORS[idx_t]}: {matrix[idx_t, idx]:.4f}" for idx_t in top_indices]
        print(f"  Top inputs: {', '.join(top_inputs)}")
        
if __name__ == '__main__':
    generate_io_matrix()
