"""
Shared utilities for the Ukraine Economic Simulation model.
Centralizes common functions to avoid duplication across modules.
"""

import numpy as np


# =============================================================================
# SECTOR WAGE PREMIUM
# =============================================================================

def get_sector_wage_premium(s):
    """
    Returns wage premium multiplier for a sector.
    Used by both CGE and ABM engines to compute effective wages.
    """
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


# =============================================================================
# GEOGRAPHIC ENERGY ZONES
# =============================================================================

# Geographic energy zone definitions based on REAL coordinates (not alphabetical index)
# West = Western Ukraine (near EU border), Center = Central Ukraine, East = Eastern/Southern Ukraine

ENERGY_ZONES = {
    # West: Lviv, Ivano-Frankivsk, Ternopil, Chernivtsi, Khmelnytskyi, Volyn, Zakarpattia
    'West': [
        'Lviv', 'Ivano-Frankivsk', 'Ternopil', 'Chernivtsi', 'Khmelnytskyi', 'Volyn', 'Zakarpattia'
    ],
    # Center: Kyiv, Cherkasy, Zhytomyr, Rivne, Vinnytsia, Poltava, Sumy, Chernihiv, Kirovohrad
    'Center': [
        'Kyiv_City', 'Kyiv_Oblast', 'Cherkasy', 'Zhytomyr', 'Rivne', 'Vinnytsia',
        'Poltava', 'Sumy', 'Chernihiv', 'Kirovohrad'
    ],
    # East/South: Dnipro, Zaporizhzhia, Kharkiv, Odesa, Mykolaiv, Kherson,
    #             Donetsk, Luhansk, Crimea, Sevastopol
    'East': [
        'Dnipro', 'Zaporizhzhia', 'Kharkiv', 'Odesa', 'Mykolaiv', 'Kherson',
        'Donetsk', 'Luhansk', 'Crimea', 'Sevastopol'
    ]
}

# Reverse lookup: region -> zone
REGION_TO_ZONE = {}
for zone, regions in ENERGY_ZONES.items():
    for r in regions:
        REGION_TO_ZONE[r] = zone


def get_energy_zone(region_name):
    """
    Returns energy zone for a region based on geographic location.
    West = relatively safe, less war damage
    Center = moderate exposure
    East = highest war exposure and cascade risk
    """
    return REGION_TO_ZONE.get(region_name, 'Center')


# =============================================================================
# DEPRECIATION RATES BY SECTOR
# =============================================================================

def get_depreciation_rate(sector):
    """
    Returns annual depreciation rate for a sector.
    """
    if sector in ['ITServicesExport', 'ITProductSaaS', 'Telecom', 'InternetCloud', 'Cybersecurity', 'EdTech']:
        return 0.25
    elif sector in ['ConstResidential', 'ConstCommercial', 'ConstInfrastructure', 'ConstReconstruction', 'RealEstateOps']:
        return 0.03
    elif sector in ['EnergyNuclearGen', 'EnergyNuclearFuel', 'EnergyNuclearWaste']:
        return 0.025
    elif sector.startswith('Mil') or sector in [
        'HeavyMachinery', 'TransportMachinery', 'AgriMachinery', 'ElectricalEquipment',
        'PrecisionInstruments', 'ElectronicsComponents', 'IndustrialRobots'
    ]:
        return 0.10
    elif sector in ['AgriGrain', 'AgriTechnical', 'AgriLivestock', 'Fishery', 'Forestry']:
        return 0.08
    else:
        return 0.07


def build_depreciation_vec(sectors):
    """
    Builds numpy vector of depreciation rates for all sectors.
    """
    return np.array([get_depreciation_rate(s) for s in sectors], dtype=np.float64)


# =============================================================================
# SHADOW ECONOMY DYNAMICS
# =============================================================================

def compute_shadow_economy_rate(year, scenario_name, eu_integration_progress):
    """
    Computes time-varying shadow economy share.
    
    Real-world dynamics:
    - 2024: ~40-45% of GDP (high wartime informality)
    - EU integration reduces shadow economy by:
        * Customs cooperation reduces smuggling
        * Digital tax administration improves compliance
        * Single market standards require formal channels
    - Optimistic scenario: faster reform, digital government
    - Pessimistic scenario: stagnation, high corruption keeps shadow economy large
    """
    # Base rate in 2026 (post-martial law, high but declining)
    base_rate = 0.42
    
    # Shadow economy shrinks with EU integration and reforms
    # EU integration progress (0 to 1 over 24 years)
    eu_effect = eu_integration_progress * 0.20  # up to 20pp reduction from EU integration
    
    # Scenario-specific reform speed
    if scenario_name == 'optimistic':
        reform_mult = 0.85  # faster decline
        digital_gov_bonus = 0.08  # additional reduction from digital government
    elif scenario_name == 'pessimistic':
        reform_mult = 1.10  # slower decline, may even increase in early years
        digital_gov_bonus = 0.0
    else:  # baseline
        reform_mult = 1.0
        digital_gov_bonus = 0.03
    
    # Time trend: shadow economy declines naturally as economy normalizes
    years_since_2026 = max(0, year - 2026)
    time_decay = years_since_2026 * 0.008  # ~0.8pp per year from natural normalization
    
    rate = base_rate - eu_effect - time_decay - digital_gov_bonus
    
    # Apply scenario multiplier
    rate *= reform_mult
    
    # Floor: even in best case, some informality remains (~15% floor)
    return np.clip(rate, 0.15, 0.55)


# =============================================================================
# CORRUPTION DYNAMICS
# =============================================================================

def compute_corruption_index(year, scenario_name, eu_integration_progress):
    """
    Computes time-varying corruption leakage rate for public procurement.
    
    Real-world CPI estimates for Ukraine:
    - 2024: CPI ~36/100 (high corruption)
    - EU accession requires reform: judicial independence, anti-corruption courts,
      public procurement transparency, digital services
    
    Corruption affects:
    - Government spending efficiency (reconstruction leakage)
    - Tax compliance (bribes, avoidance)
    - Business environment (informal payments)
    """
    # Base CPI corruption index (0.0 = no corruption, 1.0 = maximum)
    base_corruption = 0.30
    
    # EU integration reduces corruption via:
    # - Acquis alignment requires anti-corruption measures
    # - Rule of law conditionality in EU funds
    # - Public procurement directives (EU direct procurements are transparent)
    eu_effect = eu_integration_progress * 0.18
    
    # War effect: initial spike in corruption (emergency procurement)
    years_since_2026 = max(0, year - 2026)
    if years_since_2026 <= 2:
        war_spike = 0.05  # temporary spike in first 2 years
    else:
        war_spike = max(0, 0.05 - (years_since_2026 - 2) * 0.01)
    
    if scenario_name == 'optimistic':
        reform_bonus = 0.10  # strong anti-corruption reforms
        navu_effect = 0.05   # NAVU (National Agency on Corruption Prevention) enforcement
    elif scenario_name == 'pessimistic':
        reform_bonus = 0.0
        navu_effect = 0.0
        eu_effect *= 0.3  # EU integration is slow/delayed
    else:  # baseline
        reform_bonus = 0.04
        navu_effect = 0.02
    
    rate = base_corruption + war_spike - eu_effect - reform_bonus - navu_effect
    
    # Corruption cannot go below a floor (even EU members have ~5-10% corruption)
    return np.clip(rate, 0.05, 0.45)


# =============================================================================
# HOME BIAS DYNAMICS
# =============================================================================

def compute_home_bias(year, eu_integration_progress):
    """
    Computes home bias multiplier for trade gravity model.
    
    Home bias reflects preference for domestic goods:
    - Higher in developing/wartime economies (trust, logistics, language barriers)
    - Decreases with EU integration (single market requires level playing field,
      reduced barriers, competition)
    
    Baseline home bias ~5.0x (very high, similar to isolated economies)
    EU members typically have home bias ~2.0-3.0x
    """
    # Baseline home bias
    base_home_bias = 5.0
    
    # EU integration reduces home bias
    # As Ukraine integrates into EU single market, domestic goods face
    # competition from EU producers, consumers shift to imported goods
    eu_reduction = eu_integration_progress * 2.5  # reduces from 5.0 to ~2.5 over time
    
    # Natural economic development also reduces home bias
    years_since_2026 = max(0, year - 2026)
    development_reduction = years_since_2026 * 0.03
    
    total_reduction = eu_reduction + development_reduction
    
    home_bias = base_home_bias - total_reduction
    
    # Floor: even with full integration, some home preference remains
    return np.clip(home_bias, 2.0, 5.5)


# =============================================================================
# UNEMPLOYMENT / LABOR MARKET DISEQUILIBRIUM
# =============================================================================

def compute_unemployment_rate(labor_demand, labor_supply, war_intensity):
    """
    Computes unemployment rate from labor market excess supply.
    
    In real economies, labor markets do not clear instantly:
    - Search/matching friction (time to find jobs)
    - Skill mismatch (vacancies don't match available workers)
    - Geographic mismatch (jobs in different regions)
    
    War affects labor market via:
    - Mobilization (removes workers from labor force)
    - Destruction of jobs (businesses close in frontline regions)
    - Geographic displacement (workers migrate, jobs don't)
    """
    if labor_supply <= 0:
        return 0.0
    
    excess_supply = labor_supply - labor_demand
    rate = excess_supply / labor_supply
    
    # War adds frictional unemployment (job destruction exceeds creation)
    war_friction = war_intensity * 0.08  # up to 8pp extra from war
    
    rate += war_friction
    
    return np.clip(rate, 0.0, 0.35)  # max 35% unemployment (severe crisis)


# =============================================================================
# ADAPTIVE EXPECTATIONS FOR AGENTS
# =============================================================================

class AdaptiveExpectations:
    """
    Tracks agent expectations using adaptive (partial adjustment) mechanism.
    
    Agents form expectations about:
    - Inflation (price growth)
    - Wage growth
    - GRP growth (regional economic performance)
    
    Adaptive expectations: E_t[X_{t+1}] = lambda * X_t + (1-lambda) * E_{t-1}[X_t]
    where lambda is the adaption coefficient (0 < lambda < 1).
    Higher lambda = agents respond more quickly to new information.
    
    Real-world calibration:
    - Skilled workers (IT, finance): lambda ~0.6 (more information access)
    - Unskilled workers: lambda ~0.3 (slower to update)
    """
    
    def __init__(self, lambda_skilled=0.60, lambda_unskilled=0.30, lambda_semi=0.45):
        self.lambda_skilled = lambda_skilled
        self.lambda_semi = lambda_semi
        self.lambda_unskilled = lambda_unskilled
        
        # Expectation states
        self.expected_inflation = 0.0  # annual inflation rate expectation
        self.expected_wage_growth = 0.0  # annual wage growth expectation
        self.expected_grp_growth = {}  # per region
    
    def update(self, actual_inflation, actual_wage_growth, actual_grp_growth_by_region, labor_type='unskilled'):
        """
        Update expectations based on actual observed values.
        """
        if labor_type == 'skilled':
            lam = self.lambda_skilled
        elif labor_type == 'semi-skilled':
            lam = self.lambda_semi
        else:
            lam = self.lambda_unskilled
        
        # Adaptive update: new = lambda * actual + (1-lambda) * old
        self.expected_inflation = lam * actual_inflation + (1.0 - lam) * self.expected_inflation
        self.expected_wage_growth = lam * actual_wage_growth + (1.0 - lam) * self.expected_wage_growth
        
        for r, grp_g in actual_grp_growth_by_region.items():
            if r not in self.expected_grp_growth:
                self.expected_grp_growth[r] = 0.0
            self.expected_grp_growth[r] = lam * grp_g + (1.0 - lam) * self.expected_grp_growth[r]
    
    def get_expected_inflation(self):
        return self.expected_inflation
    
    def get_expected_wage_growth(self):
        return self.expected_wage_growth
    
    def get_expected_grp_growth(self, region):
        return self.expected_grp_growth.get(region, 0.0)
    
    def get_real_wage_expectation(self, nominal_wage):
        """
        Expected real wage = nominal_wage / (1 + expected_inflation)
        """
        return nominal_wage / (1.0 + max(-0.5, self.expected_inflation))


# =============================================================================
# HOUSING MARKET MODEL
# =============================================================================

class HousingMarket:
    """
    Simple housing market model that captures:
    - Housing wealth accumulation (agents save in housing)
    - Housing price dynamics (supply/demand, war destruction)
    - Wealth effect on consumption (housing wealth → consumption)
    
    Uses a reduced-form approach:
    - Housing supply: H = H_prev * (1 + construction_rate - destruction)
    - Housing demand: driven by population and income
    - Price: P = P_prev * (demand/supply) * (1 + inflation_expectation)
    
    War effects:
    - Frontline/occupied regions: housing destruction, population outflow
    - Reconstruction demand in liberated regions
    """
    
    def __init__(self, regions):
        self.regions = regions
        self.R = len(regions)
        
        # Housing stock per region (number of housing units, simplified to UAH value)
        self.housing_stock = {r: 1.0 for r in regions}  # normalized to 1.0
        self.housing_prices = {r: 1.0 for r in regions}  # normalized price index
        
        # Construction and destruction parameters
        self.base_construction_rate = 0.02  # 2% annual new construction
        self.destruction_rate = {r: 0.0 for r in regions}
    
    def step(self, population, income_per_capita, war_damage_by_region, frontline_states, expected_inflation):
        """
        Updates housing market for one year.
        
        Returns:
            housing_wealth_effect: additional consumption boost from housing wealth
        """
        housing_wealth_effect = {}
        
        for r in self.regions:
            state = frontline_states.get(r, 0)
            
            # Destruction from war
            war_dmg = war_damage_by_region.get(r, {})
            avg_dmg = sum(war_dmg.values()) / max(1, len(war_dmg)) if war_dmg else 0.0
            
            if state == 2:  # Occupied
                destruction = 0.30  # 30% housing destruction
            elif state == 1:  # Frontline
                destruction = 0.08  # 8% annual destruction
            else:
                destruction = avg_dmg * 0.5  # minor damage
            
            # Construction: depends on income and reconstruction activity
            if state == 1 or r in ['Donetsk', 'Luhansk', 'Kherson', 'Zaporizhzhia']:
                # Reconstruction zones
                construction = self.base_construction_rate * 2.0  # doubled reconstruction
            else:
                construction = self.base_construction_rate
            
            # Net housing stock change
            self.housing_stock[r] *= (1.0 + construction - destruction)
            self.housing_stock[r] = max(0.1, self.housing_stock[r])
            
            # Housing demand (driven by population and income)
            pop_factor = population.get(r, 1.0)
            income_factor = income_per_capita.get(r, 1.0)
            demand = pop_factor * income_factor
            
            # Price adjustment
            demand_supply_ratio = demand / max(0.01, self.housing_stock[r])
            price_change = demand_supply_ratio * (1.0 + expected_inflation)
            self.housing_prices[r] *= price_change
            self.housing_prices[r] = max(0.1, self.housing_prices[r])
            
            # Housing wealth (stock * price)
            housing_wealth = self.housing_stock[r] * self.housing_prices[r]
            
            # Wealth effect on consumption: 0.02-0.04 of housing wealth flows to consumption
            # (life-cycle / permanent income hypothesis)
            wealth_effect_coef = 0.03
            if state == 1:
                wealth_effect_coef = 0.01  # less liquid in frontline
            elif state == 2:
                wealth_effect_coef = 0.0  # no wealth effect in occupied
            
            housing_wealth_effect[r] = housing_wealth * wealth_effect_coef
        
        return housing_wealth_effect
    
    def get_housing_wealth(self, region):
        return self.housing_stock.get(region, 1.0) * self.housing_prices.get(region, 1.0)