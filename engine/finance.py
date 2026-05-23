import numpy as np

class FinanceEngine:
    """
    Macroeconomic and financial submodel for UkrEcoSim2050.
    Tracks government budget, public debt, monetary variables (inflation, interest rate, exchange rate),
    and balance of payments using system dynamics equations solved via RK4-like Euler steps.
    """
    def __init__(self, initial_debt_gdp=0.85, initial_exchange_rate=40.0, initial_money_supply=2.0e12):
        """
        initial_debt_gdp: public debt as % of GDP (base 2026 ~85-90%)
        initial_exchange_rate: UAH per USD
        initial_money_supply: base money supply in UAH
        """
        self.debt_gdp = initial_debt_gdp
        self.exchange_rate = initial_exchange_rate # UAH / USD
        self.money_supply = initial_money_supply # UAH
        
        self.inflation_rate = 0.08 # 8% initial inflation
        self.interest_rate = 0.15 # NBU key policy rate 15%
        
        # Policy targets
        self.inflation_target = 0.05
        
        # Fiscal tax rates
        self.tax_pit = 0.18 # Personal Income Tax
        self.tax_cit = 0.18 # Corporate Income Tax
        self.tax_vat = 0.20 # VAT
        self.tax_military = 0.05 # Military levy (5%)

    def step(self, year, nominal_gdp_uah, nominal_gdp_usd, total_wages, corporate_profits, exports_usd, imports_usd, scenario_modifiers):
        """
        Executes one year fiscal and monetary transition.
        
        Outputs:
            - state budget details (revenue, expenditure, deficit)
            - debt levels
            - monetary indices (inflation, interest rate, exchange rate)
            - trade balance
        """
        # Scenario parameters
        defense_spend_ratio = scenario_modifiers.get('defense_spending_ratio', 0.25) # % of GDP on defense
        social_spend_ratio = scenario_modifiers.get('social_spending_ratio', 0.15) # % of GDP on social/health/edu
        reconstruction_needs = scenario_modifiers.get('reconstruction_needs_usd', 15.0e9) # Reconstruction needs per year
        foreign_aid_usd = scenario_modifiers.get('foreign_aid_usd', 25.0e9) # Grants & concessional loans
        grant_share = scenario_modifiers.get('foreign_aid_grant_share', 0.60) # What share of aid is grants (non-debt)
        fdi_usd = scenario_modifiers.get('fdi_usd', 3.0e9) # Foreign Direct Investment
        
        # 1. Tax Revenues (UAH)
        pit_rev = total_wages * (self.tax_pit + self.tax_military)
        cit_rev = corporate_profits * self.tax_cit
        vat_rev = nominal_gdp_uah * 0.5 * self.tax_vat # Assuming 50% of GDP is consumption subject to VAT
        customs_rev = imports_usd * self.exchange_rate * 0.05 # 5% average tariff
        
        total_tax_revenue = pit_rev + cit_rev + vat_rev + customs_rev
        
        # 2. Expenditures (UAH)
        defense_exp = nominal_gdp_uah * defense_spend_ratio
        social_exp = nominal_gdp_uah * social_spend_ratio
        
        # Reconstruction expenditures (UAH)
        # Sourced by foreign aid + state budget contribution
        recon_exp = reconstruction_needs * self.exchange_rate
        
        # Debt servicing costs (UAH)
        # Average interest rate on national debt is modeled as a blend of domestic & external rates (~6% blended)
        blended_debt_rate = 0.065
        debt_uah = self.debt_gdp * nominal_gdp_uah
        debt_service = debt_uah * blended_debt_rate
        
        total_expenditure = defense_exp + social_exp + recon_exp + debt_service
        
        # 3. Budget Deficit & Foreign Aid (UAH)
        budget_deficit = total_expenditure - total_tax_revenue
        
        # Convert aid to UAH
        aid_uah = foreign_aid_usd * self.exchange_rate
        aid_grants_uah = aid_uah * grant_share
        aid_loans_uah = aid_uah * (1.0 - grant_share)
        
        # Net financing gap (deficit minus grants)
        financing_gap = budget_deficit - aid_grants_uah
        
        # 4. Debt Accumulation Dynamics
        # New debt is accumulated through loans (aid loans + domestic borrowing to cover financing gap)
        new_debt_uah = aid_loans_uah + max(0.0, financing_gap)
        
        next_debt_uah = debt_uah + new_debt_uah
        
        # Update Debt-to-GDP ratio (using new nominal GDP)
        self.debt_gdp = next_debt_uah / nominal_gdp_uah
        
        # 5. Monetary dynamics (Exchange Rate & Inflation)
        # Balance of Payments: Capital Account (Aid + FDI) + Current Account (Exports - Imports)
        trade_balance_usd = exports_usd - imports_usd
        bop_usd = trade_balance_usd + foreign_aid_usd + fdi_usd
        
        # Exchange rate shift: positive BOP strengthens UAH (decreases exchange rate), deficit weakens it
        # Sensitivity factor kappa
        kappa = 0.15
        er_pct_change = -kappa * (bop_usd / nominal_gdp_usd)
        # Limit annual exchange rate change to protect numerical stability (max -15% appreciation, +40% depreciation)
        er_pct_change = np.clip(er_pct_change, -0.15, 0.40)
        self.exchange_rate *= (1.0 + er_pct_change)
        
        # Money supply growth from deficit monetization (if financing gap > 0, part is monetized by NBU buying bonds)
        monetization_rate = scenario_modifiers.get('deficit_monetization_rate', 0.10) # 10% of financing gap monetized
        monetized_amount = max(0.0, financing_gap) * monetization_rate
        
        prev_money_supply = self.money_supply
        self.money_supply += monetized_amount
        ms_growth = (self.money_supply - prev_money_supply) / prev_money_supply
        
        # Inflation rate dynamics (dependent on money supply growth and GDP growth)
        gdp_growth = scenario_modifiers.get('gdp_growth', 0.03)
        theta = 0.5 # Monetarist coefficient
        
        # RK4-like Euler integration step for inflation:
        # dinflation/dt = theta * (money_growth - gdp_growth) - policy_tightness * (interest_rate - inflation)
        policy_tightness = 0.25
        d_inflation = theta * (ms_growth - gdp_growth) - policy_tightness * (self.interest_rate - self.inflation_rate)
        self.inflation_rate = max(0.01, self.inflation_rate + d_inflation)
        
        # NBU Interest Rate Policy Rule (Taylor-like response to inflation)
        # Key rate = neutral_rate + 1.5 * (inflation - inflation_target)
        neutral_rate = 0.05
        self.interest_rate = max(0.02, neutral_rate + 1.5 * (self.inflation_rate - self.inflation_target))
        
        return {
            'tax_revenue_uah': total_tax_revenue,
            'expenditure_uah': total_expenditure,
            'deficit_uah': budget_deficit,
            'deficit_gdp': budget_deficit / nominal_gdp_uah,
            'debt_gdp': self.debt_gdp,
            'exchange_rate': self.exchange_rate,
            'inflation_rate': self.inflation_rate,
            'interest_rate': self.interest_rate,
            'trade_balance_usd': trade_balance_usd,
            'bop_usd': bop_usd,
            'money_supply': self.money_supply
        }
