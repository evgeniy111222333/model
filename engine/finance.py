import numpy as np

class FinanceEngine:
    """
    Macroeconomic, banking, and financial system engine.
    Manages commercial banking balance sheets, credit creation, government deficit financing,
    OVDP bond yield curves, and NBU monetary policy rules.
    """
    def __init__(self, initial_debt_gdp=0.85, initial_exchange_rate=40.0, initial_money_supply=2.0e12):
        """
        initial_debt_gdp: public debt as % of GDP
        initial_exchange_rate: UAH per USD
        initial_money_supply: base money supply in UAH
        """
        self.debt_gdp = initial_debt_gdp
        self.exchange_rate = initial_exchange_rate # UAH / USD
        self.money_supply = initial_money_supply # UAH (Base Money)
        
        self.inflation_rate = 0.08 # 8% initial inflation
        self.interest_rate = 0.15 # NBU key policy rate 15%
        self.inflation_target = 0.05
        
        # Fiscal tax rates
        self.tax_pit = 0.18 # Personal Income Tax
        self.tax_cit = 0.18 # Corporate Income Tax
        self.tax_vat = 0.20 # VAT
        self.tax_military = 0.05 # Military levy (5%)
        
        # Commercial Banking Balance Sheet
        self.reserve_ratio = 0.10 # NBU reserve requirement (10%)
        self.deposits = 1.8e12 # Household savings deposits (UAH)
        self.reserves = self.deposits * self.reserve_ratio
        
        # Assets: Reserves + Loans + OVDP_Bonds = Deposits + Equity
        self.equity = 0.15e12
        self.loans = 0.9e12
        self.ovdp_bonds = self.deposits + self.equity - self.reserves - self.loans
        
        # Government Debt Structure (Domestic vs External)
        self.external_debt_usd = 85.0e9 # External public debt in USD
        self.domestic_debt_uah = self.ovdp_bonds # Domestic debt held by banks

    def get_yield_curve(self):
        """
        Returns the dynamic yield curve for Ukrainian government bonds (OVDP)
        for 1-year, 3-year, and 5-year durations.
        """
        inflation_exp = 0.7 * self.inflation_rate + 0.3 * self.inflation_target
        theta = 0.8
        
        yields = {}
        for term, term_premium in [('1Y', 0.015), ('3Y', 0.035), ('5Y', 0.055)]:
            yields[term] = self.interest_rate + term_premium + theta * (inflation_exp - self.inflation_target)
            yields[term] = max(0.02, yields[term]) # Floor at 2% nominal yield
        return yields

    def step(self, year, nominal_gdp_uah, nominal_gdp_usd, total_wages, corporate_profits, exports_usd, imports_usd, scenario_modifiers, household_wealth_sum=None):
        """
        Executes one year fiscal and monetary transition.
        """
        # Sync deposits to actual household wealth if provided by the model runner
        if household_wealth_sum is not None:
            self.deposits = max(0.1e12, household_wealth_sum)
            self.reserves = self.deposits * self.reserve_ratio
            
        # Scenario parameters
        defense_spend_ratio = scenario_modifiers.get('defense_spending_ratio', 0.25)
        social_spend_ratio = scenario_modifiers.get('social_spending_ratio', 0.15)
        reconstruction_needs = scenario_modifiers.get('reconstruction_needs_usd', 15.0e9)
        foreign_aid_usd = scenario_modifiers.get('foreign_aid_usd', 22.0e9)
        grant_share = scenario_modifiers.get('foreign_aid_grant_share', 0.50)
        fdi_usd = scenario_modifiers.get('fdi_usd', 1.5e9)
        
        # 1. Tax Revenues (UAH)
        pit_rev = total_wages * (self.tax_pit + self.tax_military)
        cit_rev = corporate_profits * self.tax_cit
        vat_rev = nominal_gdp_uah * 0.5 * self.tax_vat
        customs_rev = imports_usd * self.exchange_rate * 0.05
        
        total_tax_revenue = pit_rev + cit_rev + vat_rev + customs_rev
        
        # 2. Expenditures (UAH)
        defense_exp = nominal_gdp_uah * defense_spend_ratio
        social_exp = nominal_gdp_uah * social_spend_ratio
        recon_exp = reconstruction_needs * self.exchange_rate
        
        # Debt service costs: External (at 3.5% average rate) + Domestic (at 1-year OVDP yield)
        yields = self.get_yield_curve()
        domestic_yield = yields['1Y']
        
        external_debt_uah = self.external_debt_usd * self.exchange_rate
        debt_service_ext = external_debt_uah * 0.035
        debt_service_dom = self.domestic_debt_uah * domestic_yield
        debt_service = debt_service_ext + debt_service_dom
        
        total_expenditure = defense_exp + social_exp + recon_exp + debt_service
        
        # 3. Budget Deficit & Financing
        budget_deficit = total_expenditure - total_tax_revenue
        
        # Aid in UAH
        aid_uah = foreign_aid_usd * self.exchange_rate
        aid_grants_uah = aid_uah * grant_share
        aid_loans_usd = foreign_aid_usd * (1.0 - grant_share)
        
        # Net financing gap (covered by domestic borrowing and monetization)
        domestic_financing_need = budget_deficit - aid_grants_uah
        
        # Monetization rate (NBU buying bonds directly)
        monetization_rate = scenario_modifiers.get('deficit_monetization_rate', 0.08)
        monetized_amount = max(0.0, domestic_financing_need) * monetization_rate
        domestic_borrowing = max(0.0, domestic_financing_need) - monetized_amount
        
        # 4. Debt Accumulation & Bank Balance Sheet Updates
        # External debt growth
        self.external_debt_usd += aid_loans_usd
        # Domestic debt growth
        self.domestic_debt_uah += domestic_borrowing
        
        total_debt_uah = (self.external_debt_usd * self.exchange_rate) + self.domestic_debt_uah
        self.debt_gdp = total_debt_uah / nominal_gdp_uah
        
        # Commercial Bank assets updates
        # Bank absorbs domestic borrowing as new OVDP bonds
        self.ovdp_bonds = self.domestic_debt_uah
        
        # Credit Multiplier: Bank creates loans up to remaining capacity
        # Assets = Reserves (10%) + Loans + OVDP_Bonds = Deposits + Equity
        available_for_loans = self.deposits + self.equity - self.reserves - self.ovdp_bonds
        self.loans = max(0.1e12, available_for_loans) # Floor to prevent credit freeze
        
        # 5. Macro-Monetary & Exchange Rate dynamics
        # BOP: Trade balance + Aid + FDI + external debt flows
        trade_balance_usd = exports_usd - imports_usd
        capital_account_flows = foreign_aid_usd + fdi_usd
        bop_usd = trade_balance_usd + capital_account_flows
        
        # Exchange rate shifts (BOP surpluses strengthen UAH, deficits weaken it)
        kappa = 0.15
        er_pct_change = -kappa * (bop_usd / max(1e9, nominal_gdp_usd))
        # Limit annual fluctuations to protect numerical bounds
        er_pct_change = np.clip(er_pct_change, -0.15, 0.40)
        self.exchange_rate *= (1.0 + er_pct_change)
        
        # Money supply growth (M2) includes base money growth + bank credit expansion
        prev_money_supply = self.money_supply
        self.money_supply += monetized_amount # increase base money
        
        # Total M2 including multiplier effect
        m2_money_supply = self.money_supply / self.reserve_ratio
        
        # Base Money Growth Rate
        ms_growth = (self.money_supply - prev_money_supply) / max(1e-5, prev_money_supply)
        
        # Inflation rate dynamics (Taylor-Monetarist)
        gdp_growth = scenario_modifiers.get('gdp_growth', 0.03)
        theta_m = 0.4 # Money growth pass-through
        theta_er = 0.25 # Exchange rate pass-through to inflation
        
        d_inflation = (
            theta_m * (ms_growth - gdp_growth) +
            theta_er * max(0.0, er_pct_change) -
            0.20 * (self.interest_rate - self.inflation_rate)
        )
        self.inflation_rate = max(0.01, self.inflation_rate + d_inflation)
        
        # NBU Taylor policy rule (interest rate response to inflation deviations)
        neutral_rate = 0.05
        self.interest_rate = max(0.02, neutral_rate + 1.6 * (self.inflation_rate - self.inflation_target))
        
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
            'money_supply': m2_money_supply,
            'bank_loans': self.loans,
            'bank_deposits': self.deposits,
            'yield_curve': yields
        }
