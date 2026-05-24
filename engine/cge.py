import numpy as np
import scipy.optimize as opt

def get_sector_wage_premium(s):
    # IT sectors
    if s in ['ITServicesExport', 'ITProductSaaS', 'Cybersecurity', 'Telecom', 'InternetCloud']:
        return 3.0
    # Finance sectors
    elif s in ['BankState', 'BankCommercial', 'BankRetail', 'Insurance', 'NonBankFinance', 'SecuritiesMarket', 'InternationalFinance']:
        return 1.4
    # Defense sectors
    elif s in ['MilSmallArms', 'MilArmoredVehicles', 'MilArtillery', 'MilMissiles', 'MilUAVs', 'MilEW', 'MilNaval', 'MilProtectiveGear']:
        return 1.5
    # Energy Nuclear & utilities
    elif s in ['EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste', 'EnergyTransmission']:
        return 1.3
    # Health private/pharma
    elif s in ['HealthPrivate', 'PharmaGenerics', 'PharmaOriginals', 'PharmaAPI', 'MedicalDevices', 'Biotechnologies']:
        return 1.2
    # Agriculture
    elif s in ['AgriGrain', 'AgriTechnical', 'AgriLivestock', 'Fishery', 'Forestry']:
        return 0.7
    # Retail / Service / Tourism
    elif s in ['TradeRetail', 'TradeWholesale', 'HotelsTourism', 'FoodServices', 'Beverages', 'Tobacco']:
        return 0.8
    # Public Admin / Education / Public Healthcare
    elif s in ['PublicAdmin', 'LawEnforcement', 'UtilityServices', 'GasHeatSupply', 'MilitaryDefense', 'GeneralEduVoc', 'HigherEducation', 'HealthPublic']:
        return 0.9
    # Metallurgy, Construction, Chemicals, Machinery, Transport, others
    else:
        return 1.0

class CGESolver:
    """
    Computable General Equilibrium (CGE) market clearing solver.
    Solves commodity prices and labor wages equations using either Powell's hybrid method
    (for legacy S <= 15 models) or vectorized tatonnement iterations (for high-dimensional S = 93 models).
    """
    def __init__(self, regions, sectors, base_tech_coefficients, distances=None):
        self.regions = regions
        self.sectors = sectors
        self.R = len(regions)
        self.S = len(sectors)
        self.N = self.R * self.S
        
        self.base_tech = base_tech_coefficients
        self.distances = distances
        
        # Mapping index
        self.node_to_idx = {}
        self.idx_to_node = []
        idx = 0
        for r in self.regions:
            for s in self.sectors:
                self.node_to_idx[(r, s)] = idx
                self.idx_to_node.append((r, s))
                idx += 1
                
        # Build direct requirement coefficient matrix B (S x S)
        self.B_mat = np.zeros((self.S, self.S))
        for s_idx, s in enumerate(self.sectors):
            reqs = self.base_tech.get(s, {})
            for s_in, coeff in reqs.items():
                if s_in in self.sectors:
                    s_in_idx = self.sectors.index(s_in)
                    self.B_mat[s_in_idx, s_idx] = coeff
                
        # Elasticities of Substitution
        self.sigma_VA = 0.85   # Value Added: Labor vs Capital-Energy
        self.sigma_KE = 0.50   # Capital vs Energy
        self.sigma_L = 1.25    # Labor types: Skilled vs Semi-skilled vs Unskilled
        self.sigma_INT = 0.30  # Intermediate inputs
        self.eta_Arm = 2.20    # Armington Regional Trade
        self.eta_World = 1.80  # Armington World Import
        
        # CES Share parameters
        self.theta_u = 0.50
        self.theta_m = 0.35
        self.theta_s = 0.15
        
        # Sector-specific elasticities of production (alpha=capital, beta=labor, gamma=energy)
        self.alpha_vec = np.zeros(self.S)
        self.beta_vec = np.zeros(self.S)
        self.gamma_vec = np.zeros(self.S)
        
        for s_idx, s in enumerate(self.sectors):
            # IT / SaaS
            if s in ['ITServicesExport', 'ITProductSaaS', 'Cybersecurity', 'Telecom', 'InternetCloud', 'EdTech']:
                self.beta_vec[s_idx] = 0.85
                self.alpha_vec[s_idx] = 0.12
                self.gamma_vec[s_idx] = 0.03
            # Energy / Utilities
            elif s in ['EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste', 'EnergyThermal', 'EnergyTransmission', 'UtilityServices', 'GasHeatSupply']:
                self.alpha_vec[s_idx] = 0.70
                self.beta_vec[s_idx] = 0.15
                self.gamma_vec[s_idx] = 0.15
            # Agriculture
            elif s in ['AgriGrain', 'AgriTechnical', 'AgriLivestock', 'Fishery', 'Forestry']:
                self.alpha_vec[s_idx] = 0.40
                self.beta_vec[s_idx] = 0.40
                self.gamma_vec[s_idx] = 0.20
            # Metallurgy / Heavy Industry / Chemicals
            elif s in ['SteelIron', 'MetalProducts', 'NonFerrousMetal', 'IronOreMining', 'ChemicalFertilizers', 'IndustrialChemicals', 'PetrochemicalsPlastics', 'BuildingMaterials']:
                self.alpha_vec[s_idx] = 0.50
                self.beta_vec[s_idx] = 0.35
                self.gamma_vec[s_idx] = 0.15
            # Healthcare / PublicAdmin / Education
            elif s in ['HealthPublic', 'HealthPrivate', 'HealthRehab', 'HealthMental', 'PublicAdmin', 'LawEnforcement', 'MilitaryDefense', 'GeneralEduVoc', 'HigherEducation']:
                self.beta_vec[s_idx] = 0.75
                self.alpha_vec[s_idx] = 0.20
                self.gamma_vec[s_idx] = 0.05
            # Other sectors (Machinery, Military-Industrial, Transport, Retail, Finance, Tourism, etc.)
            else:
                self.alpha_vec[s_idx] = 0.35
                self.beta_vec[s_idx] = 0.50
                self.gamma_vec[s_idx] = 0.15
                
        # Derive nested share parameters
        self.theta_L_vec = self.beta_vec
        self.theta_KE_vec = 1.0 - self.beta_vec
        self.theta_K_vec = self.alpha_vec / np.clip(1.0 - self.beta_vec, 1e-5, None)
        self.theta_E_vec = self.gamma_vec / np.clip(1.0 - self.beta_vec, 1e-5, None)
        
        # Sector-specific depreciation rates
        self.depreciation_vec = np.zeros(self.S)
        for s_idx, s in enumerate(self.sectors):
            # IT
            if s in ['ITServicesExport', 'ITProductSaaS', 'Telecom', 'InternetCloud', 'Cybersecurity', 'EdTech']:
                self.depreciation_vec[s_idx] = 0.25
            # Buildings / Real Estate
            elif s in ['ConstResidential', 'ConstCommercial', 'ConstInfrastructure', 'ConstReconstruction', 'RealEstateOps']:
                self.depreciation_vec[s_idx] = 0.03
            # Nuclear Energy
            elif s in ['EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste']:
                self.depreciation_vec[s_idx] = 0.025
            # Machinery & Military-Industrial
            elif s in ['HeavyMachinery', 'TransportMachinery', 'AgriMachinery', 'ElectricalEquipment', 'PrecisionInstruments', 'ElectronicsComponents', 'IndustrialRobots'] or s.startswith('Mil'):
                self.depreciation_vec[s_idx] = 0.10
            # Agriculture
            elif s in ['AgriGrain', 'AgriTechnical', 'AgriLivestock', 'Fishery', 'Forestry']:
                self.depreciation_vec[s_idx] = 0.08
            # Others
            else:
                self.depreciation_vec[s_idx] = 0.07
                
        self.wage_premium_vec = np.array([get_sector_wage_premium(s) for s in self.sectors], dtype=np.float64)
        
        self.calibrated = False
        self.trade_shares = None
        
        # Determine energy sector indices and weights for aggregate energy price index
        self.energy_indices = []
        self.energy_weights = []
        
        if self.S <= 15:
            if 'Energy' in self.sectors:
                self.energy_indices.append(self.sectors.index('Energy'))
                self.energy_weights.append(1.0)
            else:
                self.energy_indices.append(0)
                self.energy_weights.append(1.0)
        else:
            # Energy sub-sectors for 93-sector model
            energy_subsectors = [
                'EnergyThermal', 'EnergyNuclearGen', 'EnergyNuclearFuel', 
                'EnergyNuclearWaste', 'EnergySolar', 'EnergyWindHydro', 
                'EnergyTransmission', 'CoalMining', 'OilGasExtraction', 'GasHeatSupply'
            ]
            weights = {
                'CoalMining': 0.08, 'OilGasExtraction': 0.12,
                'EnergyThermal': 0.20, 'EnergyNuclearGen': 0.25, 'EnergyNuclearFuel': 0.05,
                'EnergyNuclearWaste': 0.03, 'EnergySolar': 0.07, 'EnergyWindHydro': 0.05,
                'EnergyTransmission': 0.10, 'GasHeatSupply': 0.05
            }
            tot_w = sum(weights.get(s, 0.0) for s in energy_subsectors if s in self.sectors)
            for s in energy_subsectors:
                if s in self.sectors:
                    idx = self.sectors.index(s)
                    self.energy_indices.append(idx)
                    self.energy_weights.append(weights.get(s, 1.0) / (tot_w if tot_w > 0 else 1.0))
                    
        self.energy_indices = np.array(self.energy_indices, dtype=np.int32)
        self.energy_weights = np.array(self.energy_weights, dtype=np.float64)

        # Base import and export shares for 93-sector model
        self.base_export_shares = {
            'AgriGrain': 0.35, 'AgriTechnical': 0.30, 'SteelIron': 0.35, 'MetalProducts': 0.25, 
            'NonFerrousMetal': 0.20, 'ChemicalFertilizers': 0.15, 'ITServicesExport': 0.55, 
            'ITProductSaaS': 0.20, 'HeavyMachinery': 0.12, 'TransportMachinery': 0.10,
            'AgriMachinery': 0.08, 'ElectronicsComponents': 0.10, 'FoodProcessing': 0.15,
            'TextilesApparel': 0.10, 'LeatherFootwear': 0.08, 'FurnitureHome': 0.08,
            'PetrochemicalsPlastics': 0.10, 'IndustrialChemicals': 0.08, 'Tourism': 0.05,
            'Beverages': 0.05, 'HotelsTourism': 0.05
        }
        self.base_import_shares = {
            'CoalMining': 0.10, 'OilGasExtraction': 0.25, 'EnergyThermal': 0.05, 'EnergyNuclearFuel': 0.40,
            'ChemicalFertilizers': 0.15, 'IndustrialChemicals': 0.15, 'PetrochemicalsPlastics': 0.20,
            'BuildingMaterials': 0.05, 'PulpPaper': 0.15, 'PharmaAPI': 0.40, 'PharmaGenerics': 0.20,
            'PharmaOriginals': 0.35, 'MedicalDevices': 0.30, 'HeavyMachinery': 0.25,
            'TransportMachinery': 0.22, 'AgriMachinery': 0.20, 'ElectricalEquipment': 0.18,
            'PrecisionInstruments': 0.25, 'ElectronicsComponents': 0.35, 'IndustrialRobots': 0.30,
            'MilSmallArms': 0.20, 'MilArmoredVehicles': 0.35, 'MilArtillery': 0.30, 'MilMissiles': 0.35,
            'MilUAVs': 0.25, 'MilEW': 0.25, 'MilNaval': 0.35, 'MilProtectiveGear': 0.15,
            'FoodProcessing': 0.10, 'TextilesApparel': 0.22, 'LeatherFootwear': 0.18, 'FurnitureHome': 0.10,
            'ITProductSaaS': 0.15, 'Telecom': 0.08, 'InternetCloud': 0.10, 'Cybersecurity': 0.12
        }
        
        self.base_export_vec = np.array([self.base_export_shares.get(s, 0.01) for s in self.sectors])
        self.base_import_vec = np.array([self.base_import_shares.get(s, 0.05) for s in self.sectors])

        self.realized_imports = {}
        self.realized_exports = {}
        
        # Markup rates by sector (from FirmAgent defaults + sector specifics)
        self.sector_markup = np.zeros(self.S)
        for s_idx, s in enumerate(self.sectors):
            # Base markup of 15%, adjusted by sector
            base_markup = 0.15
            if s in ['ITServicesExport', 'ITProductSaaS', 'Cybersecurity', 'EdTech']:
                base_markup = 0.35  # High tech premium
            elif s in ['EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste']:
                base_markup = 0.12  # Regulated utilities
            elif s in ['AgriGrain', 'AgriTechnical', 'AgriLivestock']:
                base_markup = 0.08  # Competitive agriculture
            elif s in ['TradeRetail', 'TradeWholesale', 'FoodServices']:
                base_markup = 0.12  # Trade margin
            elif s in ['RealEstateOps', 'BankState', 'BankCommercial', 'BankRetail', 'NonBankFinance', 'SecuritiesMarket', 'InternationalFinance']:
                base_markup = 0.20  # Financial services
            elif s == 'Insurance':
                base_markup = 0.18  # Insurance margin
            elif s.startswith('Mil'):
                base_markup = 0.10  # Defense contracts
            self.sector_markup[s_idx] = base_markup
        
        self.interest_rate = 0.15
        self.p_world_import = np.ones(self.S)
        self.p_world_export = np.ones(self.S)
        
    def _init_distances(self):
        if self.R == 27:
            try:
                from data.loader import calculate_geographic_distances
                self.distances = calculate_geographic_distances()
                return
            except Exception:
                pass
        self.distances = np.zeros((self.R, self.R))
        for i in range(self.R):
            for j in range(self.R):
                if i == j:
                    self.distances[i, j] = 1.0
                else:
                    self.distances[i, j] = 100.0 + abs(i - j) * 35.0

    def calibrate_parameters(self, capital, labor_supply, tfp, target_demand):
        """
        Calibrates trade shares using gravity formulation.
        """
        if self.distances is None:
            self._init_distances()
            
        self.trade_shares = np.zeros((self.R, self.R, self.S))
        
        for s_idx, s in enumerate(self.sectors):
            for r_dest_idx in range(self.R):
                weights = np.zeros(self.R)
                for r_src_idx in range(self.R):
                    dist = self.distances[r_src_idx, r_dest_idx]
                    cap_src = sum(capital[self.regions[r_src_idx]].values())
                    weights[r_src_idx] = cap_src / (dist ** 1.3)
                    
                weights[r_dest_idx] *= 5.0 # high home bias
                
                sum_w = np.sum(weights)
                if sum_w > 0:
                    weights /= sum_w
                else:
                    weights = np.zeros(self.R)
                    weights[r_dest_idx] = 1.0
                    
                self.trade_shares[:, r_dest_idx, s_idx] = weights
                
        self.calibrated = True

    def evaluate_cge_algebra(self, prices, w_unskilled, w_semiskilled, w_skilled, capital_mat, labor_supply_mat, tfp_mat, energy_util_mat, hh_demands_mat, exchange_rate=40.0, interest_rate=None, p_world_import=None, p_world_export=None):
        if interest_rate is None:
            interest_rate = self.interest_rate
        if p_world_import is None:
            p_world_import = self.p_world_import
        if p_world_export is None:
            p_world_export = self.p_world_export
        """
        Evaluates core non-linear CGE equations: value added nesting, factor demands, trade flow clearance.
        """
        # 1. Labor cost index: w_L (R, S) using stable CES formulation
        # For sigma_L = 1.25 (>1), use log-space to avoid numerical instability
        sig_L = self.sigma_L
        
        if sig_L > 1.0:
            # Use nested CES: first combine semi and unskilled, then skilled
            # w_L = (theta_s * ws^rho + (theta_u*wu^rho + theta_m*wm^rho)^(rho/sigma_L))^(1/rho)
            # where rho = (sigma_L - 1) / sigma_L is negative for sigma_L > 1
            rho = (sig_L - 1.0) / sig_L  # negative when sig_L > 1
            wu_p = wu_eff ** rho
            wm_p = wm_eff ** rho
            ws_p = ws_eff ** rho
            
            # First combine unskilled and semi-skilled
            w_um = (self.theta_u * wu_p + self.theta_m * wm_p) ** (1.0 / rho)
            # Then combine with skilled using outer weights
            w_L_raw = (self.theta_s * ws_p + (1.0 - self.theta_s) * w_um ** rho) ** (1.0 / rho)
        else:
            # Standard CES for sigma_L <= 1
            w_L_raw = (self.theta_u ** sig_L * wu_eff**(1-sig_L) + 
                      self.theta_m ** sig_L * wm_eff**(1-sig_L) + 
                      self.theta_s ** sig_L * ws_eff**(1-sig_L)) ** (1.0/(1.0-sig_L))
        
        w_L = np.clip(w_L_raw, 1e4, 1e8)
        
        # Capital rent (R, S) with sector-specific depreciation rates
        rk = prices * (interest_rate + self.depreciation_vec[np.newaxis, :])
        
        # Energy price index in each region (R,)
        pe = np.sum(prices[:, self.energy_indices] * self.energy_weights[np.newaxis, :], axis=1)
        
        # 2. Capital-Energy index: p_KE (R, S)
        sig_KE = self.sigma_KE
        pe_bc = pe[:, np.newaxis]
        p_KE = (self.theta_K_vec[np.newaxis, :] ** sig_KE * rk**(1-sig_KE) + 
                self.theta_E_vec[np.newaxis, :] ** sig_KE * pe_bc**(1-sig_KE)) ** (1.0/(1.0-sig_KE))
        
        # 3. Value Added index: p_VA (R, S)
        sig_VA = self.sigma_VA
        p_VA = (self.theta_L_vec[np.newaxis, :] ** sig_VA * w_L**(1-sig_VA) + 
                self.theta_KE_vec[np.newaxis, :] ** sig_VA * p_KE**(1-sig_VA)) ** (1.0/(1.0-sig_VA))
        # p_VA is the unit cost of value added - should be in same range as output prices
        # If p_VA is too high relative to output prices, the price_ratio correction will be applied
        p_VA = np.clip(p_VA, 1.0, 1e6)
        
        # 4. Production Output: Y (R, S)
        # Production function: Y = TFP * K^alpha * E^gamma * (P/P_VA)^beta
        # WITH MARKUP: output price = marginal cost * (1 + markup)
        # So output price = p_VA (marginal cost of value added) * (1 + markup) / (1 - intermediate share)
        cap_factor = (capital_mat ** self.alpha_vec[np.newaxis, :]) * (energy_util_mat ** self.gamma_vec[np.newaxis, :])
        price_ratio = prices / np.clip(p_VA, 1e-2, None)
        # If price_ratio < 1, output is suppressed. Compensate by scaling TFP up.
        # The correct formulation normalizes by p_VA which includes labor costs.
        # For now, cap the price_ratio effect to prevent extreme suppression
        price_ratio_capped = np.clip(price_ratio, 0.1, 10.0)
        y_val = tfp_mat * cap_factor * (price_ratio_capped ** self.beta_vec[np.newaxis, :])
        
        # 5. Factor Demands
        L_composite = y_val * (p_VA / np.clip(w_L, 1e-2, None)) ** sig_VA * self.theta_L_vec[np.newaxis, :]
        
        L_u = L_composite * (w_L / np.clip(wu_eff, 1e-2, None)) ** sig_L * self.theta_u
        L_m = L_composite * (w_L / np.clip(wm_eff, 1e-2, None)) ** sig_L * self.theta_m
        L_s = L_composite * (w_L / np.clip(ws_eff, 1e-2, None)) ** sig_L * self.theta_s
        
        dem_unskilled = np.sum(L_u, axis=1)
        dem_semiskilled = np.sum(L_m, axis=1)
        dem_skilled = np.sum(L_s, axis=1)
        
        # 6. Intermediate Demands (R, S)
        intermediate_demands = y_val @ self.B_mat.T
        
        d_total = hh_demands_mat + intermediate_demands
        
        # 7. Imports and Domestic Demand
        if self.S <= 15:
            # Legacy mode
            d_domestic = d_total * 0.85
            imports = d_total * 0.15
            exports = y_val * 0.15
        else:
            # High-fidelity endogenous trade
            # Composite price index in each region (R, S)
            p_comp_mat = np.zeros((self.R, self.S))
            for s_idx in range(self.S):
                shares = self.trade_shares[:, :, s_idx]
                p_src = prices[:, s_idx]
                p_comp_mat[:, s_idx] = shares.T @ p_src
                
            if p_world_import is None:
                p_world_import_vec = np.ones(self.S)
            else:
                p_world_import_vec = np.array(p_world_import)
                
            p_import_uah = exchange_rate * p_world_import_vec[np.newaxis, :]
            import_ratio = p_comp_mat / np.clip(p_import_uah, 1e-2, None)
            imports = d_total * self.base_import_vec[np.newaxis, :] * (import_ratio ** self.eta_World)
            imports = np.clip(imports, 0.0, d_total * 0.90)
            d_domestic = d_total - imports
            
            # Exports: apply markup to export price
            # Export price in USD = domestic price * (1 + markup) / exchange_rate
            if p_world_export is None:
                p_world_export_vec = np.ones(self.S)
            else:
                p_world_export_vec = np.array(p_world_export)
                
            p_export_uah = exchange_rate * p_world_export_vec[np.newaxis, :]
            markup_vec = (1.0 + self.sector_markup[np.newaxis, :])
            export_ratio = (p_export_uah * markup_vec) / np.clip(prices, 1e-2, None)
            exports = y_val * self.base_export_vec[np.newaxis, :] * (export_ratio ** 1.8)
            exports = np.clip(exports, 0.0, y_val * 0.90)

        # 8. Interregional Trade Distribution
        total_commodity_demand = np.zeros((self.R, self.S))
        
        for s_idx in range(self.S):
            shares = self.trade_shares[:, :, s_idx]
            p_src = prices[:, s_idx]
            p_comp = np.sum(shares.T @ p_src) # average proxy for trade shares correction
            
            shares_adj = shares * (p_comp / np.clip(p_src[:, np.newaxis], 1e-2, None)) ** self.eta_Arm
            
            sum_shares = np.sum(shares_adj, axis=0)
            sum_shares[sum_shares == 0] = 1.0
            shares_adj /= sum_shares
            
            total_commodity_demand[:, s_idx] = shares_adj @ d_domestic[:, s_idx]
            
        # 9. Excess Demands
        if self.S <= 15:
            excess_commodity = (total_commodity_demand - y_val).flatten()
        else:
            # Production must equal domestic interregional demand + exports
            excess_commodity = (total_commodity_demand + exports - y_val).flatten()
            
        excess_labor_skilled = dem_skilled - labor_supply_mat[:, 2]
        excess_labor_semiskilled = dem_semiskilled - labor_supply_mat[:, 1]
        excess_labor_unskilled = dem_unskilled - labor_supply_mat[:, 0]
        
        excess = np.concatenate([
            excess_commodity, 
            excess_labor_skilled, 
            excess_labor_semiskilled, 
            excess_labor_unskilled
        ])
        
        return excess, y_val, imports, exports

    def evaluate_cge_equations(self, multipliers, capital_mat, labor_supply_mat, tfp_mat, prices_base_mat, wages_base_mat, energy_util_mat, hh_demands_mat, exchange_rate=40.0, interest_rate=None, p_world_import=None, p_world_export=None):
        if interest_rate is None:
            interest_rate = self.interest_rate
        if p_world_import is None:
            p_world_import = self.p_world_import
        if p_world_export is None:
            p_world_export = self.p_world_export
            
        mult_clip = np.clip(multipliers, 1e-3, 1e9)
        prices_mult = mult_clip[0:self.N].reshape((self.R, self.S))
        wages_skilled_mult = mult_clip[self.N:self.N+self.R]
        wages_semiskilled_mult = mult_clip[self.N+self.R:self.N+2*self.R]
        wages_unskilled_mult = mult_clip[self.N+2*self.R:self.N+3*self.R]
        
        prices = prices_base_mat * prices_mult
        w_unskilled = wages_base_mat[:, 0] * wages_unskilled_mult
        w_semiskilled = wages_base_mat[:, 1] * wages_semiskilled_mult
        w_skilled = wages_base_mat[:, 2] * wages_skilled_mult
        
        excess, _, _, _ = self.evaluate_cge_algebra(
            prices=prices,
            w_unskilled=w_unskilled,
            w_semiskilled=w_semiskilled,
            w_skilled=w_skilled,
            capital_mat=capital_mat,
            labor_supply_mat=labor_supply_mat,
            tfp_mat=tfp_mat,
            energy_util_mat=energy_util_mat,
            hh_demands_mat=hh_demands_mat,
            exchange_rate=exchange_rate,
            interest_rate=interest_rate,
            p_world_import=p_world_import,
            p_world_export=p_world_export
        )
        return excess

    def solve_equilibrium(self, capital, labor_supply_by_type, tfp, prices_init, energy_utilization, household_demands, exchange_rate=40.0, interest_rate=None, p_world_import=None, p_world_export=None):
        if interest_rate is not None:
            self.interest_rate = interest_rate
        else:
            interest_rate = self.interest_rate
            
        if p_world_import is not None:
            self.p_world_import = p_world_import
        else:
            p_world_import = self.p_world_import
            
        if p_world_export is not None:
            self.p_world_export = p_world_export
        else:
            p_world_export = self.p_world_export
            
        """
        Solves for the prices and wages that clear all CGE markets.
        Uses Powell hybrid method for small systems and fast tatonnement iterations for large systems.
        """
        if not self.calibrated:
            self.calibrate_parameters(capital, labor_supply_by_type, tfp, household_demands)
            
        capital_mat = np.zeros((self.R, self.S))
        tfp_mat = np.zeros((self.R, self.S))
        prices_init_mat = np.zeros((self.R, self.S))
        energy_util_mat = np.zeros((self.R, self.S))
        hh_demands_mat = np.zeros((self.R, self.S))
        
        for r_idx, r in enumerate(self.regions):
            for s_idx, s in enumerate(self.sectors):
                capital_mat[r_idx, s_idx] = capital[r][s]
                tfp_mat[r_idx, s_idx] = tfp[r][s]
                prices_init_mat[r_idx, s_idx] = prices_init[r][s]
                energy_util_mat[r_idx, s_idx] = energy_utilization[r][s]
                hh_demands_mat[r_idx, s_idx] = household_demands[r][s]
                
        labor_supply_mat = np.zeros((self.R, 3))
        wages_base_mat = np.zeros((self.R, 3))
        
        for r_idx, r in enumerate(self.regions):
            labor_supply_mat[r_idx, 0] = labor_supply_by_type[r]['unskilled']
            labor_supply_mat[r_idx, 1] = labor_supply_by_type[r].get('semi-skilled', labor_supply_by_type[r]['unskilled'] * 0.70)
            labor_supply_mat[r_idx, 2] = labor_supply_by_type[r]['skilled']
            
            # Get wages from scenario modifiers or use defaults from wages_by_type
            # labor_supply_by_type contains keys 'unskilled', 'semi-skilled', 'skilled' (counts)
            # We need actual wages - these should be passed or extracted from the model
            # For now, use the wages_base_mat passed from the model runner's wages_by_type
            # Note: this function receives wages_base_mat as parameter, so this block is for fallback
            w_u = wages_base_mat[r_idx, 0] if r_idx < wages_base_mat.shape[0] else 120000.0
            w_s = wages_base_mat[r_idx, 2] if r_idx < wages_base_mat.shape[0] else 300000.0
            wages_base_mat[r_idx, 0] = w_u
            wages_base_mat[r_idx, 2] = w_s
            wages_base_mat[r_idx, 1] = (w_u + w_s) / 2.0
 
        if self.S <= 15:
            # ----------------------------------------------------
            # POWELL HYBRID SOLVER (for legacy/unit tests)
            # ----------------------------------------------------
            guess = np.ones(self.N + 3 * self.R)
            
            def obj_func(vars):
                return self.evaluate_cge_equations(
                    multipliers=vars,
                    capital_mat=capital_mat,
                    labor_supply_mat=labor_supply_mat,
                    tfp_mat=tfp_mat,
                    prices_base_mat=prices_init_mat,
                    wages_base_mat=wages_base_mat,
                    energy_util_mat=energy_util_mat,
                    hh_demands_mat=hh_demands_mat,
                    exchange_rate=exchange_rate,
                    interest_rate=interest_rate,
                    p_world_import=p_world_import,
                    p_world_export=p_world_export
                )
                
            res = opt.root(obj_func, guess, method='hybr', options={'xtol': 1e-4, 'maxfev': 150})
            
            solved_mult = np.clip(res.x, 1e-3, 1e9)
            prices_mult = solved_mult[0:self.N].reshape((self.R, self.S))
            wages_skilled_mult = solved_mult[self.N:self.N+self.R]
            wages_semiskilled_mult = solved_mult[self.N+self.R:self.N+2*self.R]
            wages_unskilled_mult = solved_mult[self.N+2*self.R:self.N+3*self.R]
            
            prices_solved_mat = prices_init_mat * prices_mult
            wu_solved = wages_base_mat[:, 0] * wages_unskilled_mult
            wm_solved = wages_base_mat[:, 1] * wages_semiskilled_mult
            ws_solved = wages_base_mat[:, 2] * wages_skilled_mult
        else:
            # ----------------------------------------------------
            # VECTORIZED TATONNEMENT SOLVER (for high-dimensional 93 sectors)
            # ----------------------------------------------------
            prices_solved_mat = prices_init_mat.copy()
            wages_solved_mat = wages_base_mat.copy()
            
            gamma_p = 0.08
            gamma_w = 0.08
            
            max_iter = 100
            tol = 5e-3
            
            for it in range(max_iter):
                excess, _, _, _ = self.evaluate_cge_algebra(
                    prices=prices_solved_mat,
                    w_unskilled=wages_solved_mat[:, 0],
                    w_semiskilled=wages_solved_mat[:, 1],
                    w_skilled=wages_solved_mat[:, 2],
                    capital_mat=capital_mat,
                    labor_supply_mat=labor_supply_mat,
                    tfp_mat=tfp_mat,
                    energy_util_mat=energy_util_mat,
                    hh_demands_mat=hh_demands_mat,
                    exchange_rate=exchange_rate,
                    interest_rate=interest_rate,
                    p_world_import=p_world_import,
                    p_world_export=p_world_export
                )
                
                excess_commodity = excess[0:self.N].reshape((self.R, self.S))
                excess_skilled = excess[self.N:self.N+self.R]
                excess_semiskilled = excess[self.N+self.R:self.N+2*self.R]
                excess_unskilled = excess[self.N+2*self.R:self.N+3*self.R]
                
                max_err = np.max(np.abs(excess))
                if max_err < tol:
                    break
                    
                # Adjust prices based on relative excess demand
                # Scale step size dynamically to preserve stability
                prices_solved_mat *= (1.0 + gamma_p * np.clip(excess_commodity / np.clip(hh_demands_mat + 1.0, 1e-1, None), -0.2, 0.2))
                prices_solved_mat = np.clip(prices_solved_mat, 1e-2, 1e2)
                
                # Adjust wages based on labor excess supply/demand
                wages_solved_mat[:, 2] *= (1.0 + gamma_w * np.clip(excess_skilled / np.clip(labor_supply_mat[:, 2], 1.0, None), -0.2, 0.2))
                wages_solved_mat[:, 1] *= (1.0 + gamma_w * np.clip(excess_semiskilled / np.clip(labor_supply_mat[:, 1], 1.0, None), -0.2, 0.2))
                wages_solved_mat[:, 0] *= (1.0 + gamma_w * np.clip(excess_unskilled / np.clip(labor_supply_mat[:, 0], 1.0, None), -0.2, 0.2))
                wages_solved_mat = np.clip(wages_solved_mat, 1e2, 1e7)
                
            wu_solved = wages_solved_mat[:, 0]
            wm_solved = wages_solved_mat[:, 1]
            ws_solved = wages_solved_mat[:, 2]
 
        # Re-evaluate final outputs, imports and exports
        _, y_val, imports, exports = self.evaluate_cge_algebra(
            prices=prices_solved_mat,
            w_unskilled=wu_solved,
            w_semiskilled=wm_solved,
            w_skilled=ws_solved,
            capital_mat=capital_mat,
            labor_supply_mat=labor_supply_mat,
            tfp_mat=tfp_mat,
            energy_util_mat=energy_util_mat,
            hh_demands_mat=hh_demands_mat,
            exchange_rate=exchange_rate,
            interest_rate=interest_rate,
            p_world_import=p_world_import,
            p_world_export=p_world_export
        )
        
        # Populate results
        prices_solved = {}
        for r_idx, r in enumerate(self.regions):
            prices_solved[r] = {}
            for s_idx, s in enumerate(self.sectors):
                prices_solved[r][s] = prices_solved_mat[r_idx, s_idx]
                
        wages_solved = {}
        for r_idx, r in enumerate(self.regions):
            wages_solved[r] = {
                'unskilled': wu_solved[r_idx],
                'semi-skilled': wm_solved[r_idx],
                'skilled': ws_solved[r_idx]
            }
            
        realized_output = {}
        self.realized_imports = {}
        self.realized_exports = {}
        for r_idx, r in enumerate(self.regions):
            for s_idx, s in enumerate(self.sectors):
                realized_output[(r, s)] = max(1e-3, y_val[r_idx, s_idx])
                self.realized_imports[(r, s)] = float(imports[r_idx, s_idx])
                self.realized_exports[(r, s)] = float(exports[r_idx, s_idx])
                
        return realized_output, prices_solved, wages_solved
