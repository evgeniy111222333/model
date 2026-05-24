import numpy as np

class FinanceEngine:
    """
    Macroeconomic, banking, and financial system engine.
    Manages commercial banking balance sheets, credit creation, government deficit financing,
    OVDP bond yield curves, and NBU monetary policy rules.
    Supports a three-tier banking system (State, Commercial corporate, and Retail),
    an Insurance sector (premiums, claims, and distress feedback), and a two-pillar pension system.
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
        self.tax_pension = 0.18 # Solidarity Pension Contribution (Pillar 1)
        self.tax_pension_pillar2 = 0.04 # Accumulative Pension Contribution (Pillar 2)
        
        # NBU FX Reserves
        self.nbu_fx_reserves_usd = 38.0e9
        self.pension_rate = 50000.0
        
        # Pillar 2 Accumulative Pension Fund
        self.deposits_pension_pillar2 = 0.02e12 # UAH initial
        
        # Insurance Sector Balance Sheet
        self.insurance_equity = 0.05e12
        self.insurance_reserves = 0.20e12
        self.insurance_bonds = 0.21e12
        self.insurance_cash = 0.04e12
        
        # Three-Tier Commercial Banking Balance Sheets
        self.reserve_ratio = 0.10 # NBU reserve requirement (10%)
        
        # Retail Banks
        self.deposits_retail = 1.0e12
        self.equity_retail = 0.04e12
        self.reserves_retail = self.deposits_retail * self.reserve_ratio
        self.loans_retail = 0.4e12
        self.ovdp_bonds_retail = 0.54e12
        
        # Commercial Corporate Banks
        self.deposits_comm = 0.6e12
        self.equity_comm = 0.06e12
        self.reserves_comm = self.deposits_comm * self.reserve_ratio
        self.loans_comm = 0.5e12
        self.ovdp_bonds_comm = 0.1e12
        
        # State Banks
        self.deposits_state = 0.4e12
        self.equity_state = 0.05e12
        self.reserves_state = self.deposits_state * self.reserve_ratio
        self.ovdp_bonds_state = 0.41e12
        
        # Aggregate stats (backwards compatibility)
        self.deposits = self.deposits_retail + self.deposits_comm + self.deposits_state
        self.reserves = self.reserves_state + self.reserves_comm + self.reserves_retail
        self.loans = self.loans_comm + self.loans_retail
        self.equity = self.equity_state + self.equity_comm + self.equity_retail
        self.ovdp_bonds = self.ovdp_bonds_state + self.ovdp_bonds_comm + self.ovdp_bonds_retail
        
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

    def step(self, year, nominal_gdp_uah, nominal_gdp_usd, total_wages, corporate_profits, exports_usd, imports_usd, scenario_modifiers, household_wealth_sum=None, num_pensioners=None):
        """
        Executes one year fiscal and monetary transition.
        """
        # Sync retail deposits to actual household wealth if provided by the model runner
        if household_wealth_sum is not None:
            self.deposits_retail = max(0.1e12, household_wealth_sum)
            
        # Scenario parameters
        defense_spend_ratio = scenario_modifiers.get('defense_spending_ratio', 0.25)
        social_spend_ratio = scenario_modifiers.get('social_spending_ratio', 0.15)
        reconstruction_needs = scenario_modifiers.get('reconstruction_needs_usd', 15.0e9)
        foreign_aid_usd = scenario_modifiers.get('foreign_aid_usd', 22.0e9)
        grant_share = scenario_modifiers.get('foreign_aid_grant_share', 0.50)
        fdi_usd = scenario_modifiers.get('fdi_usd', 1.5e9)
        war_intensity = scenario_modifiers.get('war_damage_intensity', 0.02)
        
        # 1. Tax Revenues (UAH)
        pit_rev = total_wages * (self.tax_pit + self.tax_military)
        cit_rev = corporate_profits * self.tax_cit
        vat_rev = nominal_gdp_uah * 0.5 * self.tax_vat
        customs_rev = imports_usd * self.exchange_rate * 0.05
        
        # Pillar 1 Solidarity Pension Tax
        pension_solidarity_rev = total_wages * self.tax_pension
        # Pillar 2 Accumulative Pension Fund Contribution (directed to accumulative deposits)
        pension_accum_rev = total_wages * self.tax_pension_pillar2
        self.deposits_pension_pillar2 += pension_accum_rev
        
        total_tax_revenue = pit_rev + cit_rev + vat_rev + customs_rev + pension_solidarity_rev
        
        # 2. Insurance Premium Collection & Claims Payouts
        # Households pay 1.5% premium on wealth, Firms pay 0.5% premium on capital stock value
        ins_prem_hh = (household_wealth_sum if household_wealth_sum is not None else 1.0e12) * 0.015
        ins_prem_firms = (nominal_gdp_uah * 2.7) * 0.005
        total_ins_premiums = ins_prem_hh + ins_prem_firms
        
        self.insurance_reserves += total_ins_premiums
        self.insurance_cash += total_ins_premiums
        
        # Claims payout: cover 40% of capital destruction due to war
        capital_destruction_uah = nominal_gdp_uah * war_intensity * 0.40
        claims_payout = capital_destruction_uah * 0.40
        
        self.insurance_reserves = max(0.01e12, self.insurance_reserves - claims_payout)
        self.insurance_cash = max(0.005e12, self.insurance_cash - claims_payout)
        
        insurance_distress = False
        if claims_payout > self.insurance_reserves:
            excess_claims = claims_payout - self.insurance_reserves
            self.insurance_equity = max(1e9, self.insurance_equity - excess_claims)
            if self.insurance_equity < 0.01e12:
                insurance_distress = True
                
        # 3. Expenditures (UAH)
        defense_exp = nominal_gdp_uah * defense_spend_ratio
        social_exp = nominal_gdp_uah * social_spend_ratio
        recon_exp = reconstruction_needs * self.exchange_rate
        
        # Dynamic pension benefits (Pillar 1 Solidarity Fund)
        if num_pensioners is not None:
            dynamic_pension = pension_solidarity_rev / max(1e5, num_pensioners)
            self.pension_rate = np.clip(dynamic_pension, 30000.0, 90000.0)
            pension_exp = self.pension_rate * num_pensioners
        else:
            self.pension_rate = 50000.0
            pension_exp = self.pension_rate * 2.0e6
            
        # Debt service costs: External (at 3.5% average rate) + Domestic (at 1-year OVDP yield)
        yields = self.get_yield_curve()
        domestic_yield = yields['1Y']
        
        external_debt_uah = self.external_debt_usd * self.exchange_rate
        debt_service_ext = external_debt_uah * 0.035
        debt_service_dom = self.domestic_debt_uah * domestic_yield
        debt_service = debt_service_ext + debt_service_dom
        
        total_expenditure = defense_exp + social_exp + recon_exp + debt_service + pension_exp
        
        # 4. Budget Deficit & Financing
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
        
        # 5. Debt Accumulation
        self.external_debt_usd += aid_loans_usd
        self.domestic_debt_uah += domestic_borrowing
        
        total_debt_uah = (self.external_debt_usd * self.exchange_rate) + self.domestic_debt_uah
        self.debt_gdp = total_debt_uah / nominal_gdp_uah
        
        # 6. Banking Sector Updates & Non-Performing Loans (NPLs)
        # NPL rates based on war intensity and high interest rates
        npl_rate_comm = np.clip(0.03 + 0.50 * war_intensity + 0.40 * max(0.0, self.interest_rate - 0.10), 0.02, 0.60)
        npl_rate_retail = np.clip(0.02 + 0.30 * war_intensity + 0.20 * max(0.0, self.interest_rate - 0.10), 0.02, 0.40)
        
        # Systemic Panic Thresholds (Banking Crisis & Capital Flight)
        capital_flight_triggered = False
        if npl_rate_comm > 0.35 or npl_rate_retail > 0.25:
            self.deposits_retail *= 0.80
            self.deposits_comm *= 0.80
            capital_flight_triggered = True
            
        # Write-offs reduce bank equity
        writeoff_comm = self.loans_comm * npl_rate_comm * 0.15
        writeoff_retail = self.loans_retail * npl_rate_retail * 0.10
        self.equity_comm = max(1e9, self.equity_comm - writeoff_comm)
        self.equity_retail = max(1e9, self.equity_retail - writeoff_retail)
        
        # Update reserves
        self.reserves_state = self.deposits_state * self.reserve_ratio
        self.reserves_comm = self.deposits_comm * self.reserve_ratio
        self.reserves_retail = self.deposits_retail * self.reserve_ratio
        
        # Update government bond holdings across tiers
        self.ovdp_bonds_state = self.domestic_debt_uah * 0.50
        self.ovdp_bonds_comm = self.domestic_debt_uah * 0.15
        self.ovdp_bonds_retail = self.domestic_debt_uah * 0.35
        
        # Update loans with Capital Adequacy Ratio limits and liquidity capacity (crowding out)
        avail_comm = self.deposits_comm + self.equity_comm - self.reserves_comm - self.ovdp_bonds_comm
        car_comm_limit = self.equity_comm / 0.08
        self.loans_comm = max(0.1e12, min(avail_comm, car_comm_limit))
        
        avail_retail = self.deposits_retail + self.equity_retail - self.reserves_retail - self.ovdp_bonds_retail
        car_retail_limit = self.equity_retail / 0.08
        self.loans_retail = max(0.1e12, min(avail_retail, car_retail_limit))
        
        # Sync aggregate stats for backward compatibility
        self.reserves = self.reserves_state + self.reserves_comm + self.reserves_retail
        self.loans = self.loans_comm + self.loans_retail
        self.deposits = self.deposits_retail + self.deposits_comm + self.deposits_state
        self.ovdp_bonds = self.ovdp_bonds_state + self.ovdp_bonds_comm + self.ovdp_bonds_retail
        self.equity = self.equity_state + self.equity_comm + self.equity_retail
        
        # 7. BOP and NBU FX Interventions
        trade_balance_usd = exports_usd - imports_usd
        capital_account_flows = foreign_aid_usd + fdi_usd
        
        # Systemic distress feedback: Insurance distress or capital flight triggers extra outflow
        if capital_flight_triggered:
            capital_account_flows -= 2.0e9 # $2B capital flight outflow
        if insurance_distress:
            capital_account_flows -= 1.0e9 # Extra $1B outflow due to insurance collapse
            
        bop_usd = trade_balance_usd + capital_account_flows
        
        # NBU defends UAH if BOP < 0, else buys reserves
        if bop_usd < 0:
            needed_intervention = -bop_usd
            actual_intervention = min(needed_intervention, self.nbu_fx_reserves_usd * 0.30)
            self.nbu_fx_reserves_usd -= actual_intervention
            remaining_bop_deficit = needed_intervention - actual_intervention
            
            # Currency crisis if reserves drop below 5B
            if self.nbu_fx_reserves_usd < 5.0e9:
                crisis_depreciation = 0.25
                kappa = 0.35
            else:
                crisis_depreciation = 0.0
                kappa = 0.15
                
            er_pct_change = kappa * (remaining_bop_deficit / max(1e9, nominal_gdp_usd)) + crisis_depreciation
        else:
            # BOP surplus: NBU buys 50% into reserves
            reserves_added = bop_usd * 0.50
            self.nbu_fx_reserves_usd += reserves_added
            remaining_bop_surplus = bop_usd - reserves_added
            
            kappa = 0.15
            er_pct_change = -kappa * (remaining_bop_surplus / max(1e9, nominal_gdp_usd))
            
        er_pct_change = np.clip(er_pct_change, -0.15, 0.60)
        self.exchange_rate *= (1.0 + er_pct_change)
        # PPP-based floor: UAH should never be cheaper than 30 UAH/USD (Ukraine's long-term range)
        # This prevents the model from "appreciating" to unrealistic levels
        self.exchange_rate = np.clip(self.exchange_rate, 30.0, 120.0)
        
        # Money supply growth (M2) includes base money growth + bank credit expansion
        prev_money_supply = self.money_supply
        self.money_supply += monetized_amount
        m2_money_supply = self.money_supply / self.reserve_ratio
        
        ms_growth = (self.money_supply - prev_money_supply) / max(1e-5, prev_money_supply)
        
        # Inflation rate dynamics (Taylor-Monetarist)
        gdp_growth = scenario_modifiers.get('gdp_growth', 0.03)
        theta_m = 0.4
        theta_er = 0.25
        
        d_inflation = (
            theta_m * (ms_growth - gdp_growth) +
            theta_er * max(0.0, er_pct_change) -
            0.20 * (self.interest_rate - self.inflation_rate)
        )
        # Remove hard 1% floor - allow deflation if conditions warrant
        self.inflation_rate = max(-0.05, self.inflation_rate + d_inflation)
        
        # NBU Taylor policy rule (interest rate response to inflation deviations)
        neutral_rate = 0.05
        ir_shock = scenario_modifiers.get('interest_rate_shock', 0.0)
        self.interest_rate = max(0.02, neutral_rate + 1.6 * (self.inflation_rate - self.inflation_target) + ir_shock)
        
        # Pillar 2 accumulative interest accumulation
        self.deposits_pension_pillar2 *= (1.0 + self.interest_rate * 0.8)
        
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
            'yield_curve': yields,
            'pension_rate': self.pension_rate,
            'nbu_fx_reserves_usd': self.nbu_fx_reserves_usd,
            'npl_comm': npl_rate_comm,
            'npl_retail': npl_rate_retail,
            'deposits_pension_pillar2': self.deposits_pension_pillar2,
            'insurance_equity': self.insurance_equity,
            'insurance_reserves': self.insurance_reserves
        }
