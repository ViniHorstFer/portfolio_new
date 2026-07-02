"""
metrics.py — Performance & risk metric calculators.

These functions are copied VERBATIM from the analyst's validated codebase
(calc_benchmarks_standalone.py and model_total_v2.ipynb). Do not refactor the
math here without re-validating against the existing app outputs.
"""

import pandas as pd
import numpy as np
from scipy import stats

def gross_up_tax_exempt_navs(fundos_info, result_df, tax_rate=0.15):
    """
    Gross up NAVs of tax-exempt funds using additive daily return adjustment.
    Preserves the actual return dynamics while adjusting cumulative performance.
    
    Parameters:
    -----------
    fundos_info : pd.DataFrame
        DataFrame with columns 'CNPJ_FUNDO', 'VL_QUOTA', and 'DT_COMPTC' (date)
    result_df : pd.DataFrame
        DataFrame with columns 'CNPJ' and 'TRIBUTAÇÃO'
    tax_rate : float
        Withholding tax rate (default 0.15 for 15%)
    
    Returns:
    --------
    pd.DataFrame
        fundos_info with adjusted VL_QUOTA for tax-exempt funds
    """
    # Get list of tax-exempt fund CNPJs
    exempt_cnpjs = result_df[result_df["Tributação"] == "Isento"]["CNPJ"].tolist()
    
    # Create a copy to avoid modifying original
    fundos_adjusted = fundos_info.copy()
    
    # Process each tax-exempt fund
    for cnpj in exempt_cnpjs:
        # Get the fund's NAV series
        fund_mask = fundos_adjusted["CNPJ_FUNDO"] == cnpj
        fund_data = fundos_adjusted[fund_mask].copy()
        
        if len(fund_data) < 2:
            # Skip if insufficient data
            continue
        
        # Sort by date to ensure proper time series
        fund_data = fund_data.sort_values("DT_COMPTC").reset_index(drop=True)
        
        # Calculate initial and final NAVs
        nav_initial = fund_data["VL_QUOTA"].iloc[0]
        nav_final = fund_data["VL_QUOTA"].iloc[-1]
        
        if nav_initial == 0 or pd.isna(nav_initial) or nav_final == 0 or pd.isna(nav_final):
            continue
        
        # Calculate net cumulative return
        R_net = (nav_final - nav_initial) / nav_initial
        
        # Calculate target gross return
        R_target = R_net / (1 - tax_rate)  # R_net / 0.85
        
        # Number of periods
        T = len(fund_data) - 1
        
        if T == 0:
            continue
        
        # Calculate average daily returns (geometric mean)
        daily_return_net = np.power(1 + R_net, 1 / T) - 1
        daily_return_target = np.power(1 + R_target, 1 / T) - 1
        
        # Calculate additive adjustment (alpha)
        alpha = daily_return_target - daily_return_net
        
        # Calculate actual daily returns from original NAV series
        daily_returns = fund_data["VL_QUOTA"].pct_change().fillna(0).values
        
        # Apply additive adjustment to each daily return
        adjusted_returns = daily_returns + alpha
        
        # Reconstruct NAV series from adjusted returns
        adjusted_navs = np.zeros(len(fund_data))
        adjusted_navs[0] = nav_initial
        
        for i in range(1, len(adjusted_navs)):
            adjusted_navs[i] = adjusted_navs[i-1] * (1 + adjusted_returns[i])
        
        # Update the dataframe
        fund_data["VL_QUOTA"] = adjusted_navs
        
        # Update the original dataframe
        fundos_adjusted.loc[fund_mask, "VL_QUOTA"] = fund_data["VL_QUOTA"].values
    
    return fundos_adjusted


def calculate_fund_metrics(df, info_df=None, cdi=None):
    """
    Calculate standard performance metrics without exposures.
    
    This function calculates returns, volatility, Sharpe ratios, drawdowns, etc.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fund data from CVM with VL_QUOTA, VL_PATRIM_LIQ, etc.
    info_df : pd.DataFrame
        Fund metadata (names, categories, etc.)
    cdi : pd.Series
        CDI returns (benchmark)
    
    Returns:
    --------
    pd.DataFrame: Performance metrics for all funds
    """
    # Remove rows where VL_QUOTA is zero or null
    df = df[(df['VL_QUOTA'] != 0) & (df['VL_QUOTA'].notna())].copy()
    
    # Handle duplicates
    df = df.reset_index()
    df = df.sort_values('NR_COTST', ascending=False)
    df = df.drop_duplicates(subset=['CNPJ_FUNDO', df.columns[0]], keep='first')
    df = df.set_index(df.columns[0])

    df = df.reset_index()
    df = df.sort_values('NR_COTST', ascending=False)
    df = df.drop_duplicates(subset=['CNPJ_FUNDO', df.columns[0]], keep='first')
    df = df.set_index(df.columns[0])
    
    results = []
    
    for cnpj in df['CNPJ_FUNDO'].unique():
        fund_data = df[df['CNPJ_FUNDO'] == cnpj].sort_index()
        
        if len(fund_data) == 0:
            continue
        
        latest_date = fund_data.index[-1]
        latest_row = fund_data.iloc[-1]
        
        result = {
            'CNPJ_FUNDO': cnpj,
            'LAST_UPDATE': latest_date,
            'VL_PATRIM_LIQ': latest_row['VL_PATRIM_LIQ'],
            'NR_COTST': latest_row['NR_COTST'],
        }
        
        # Calculate returns
        quota_series = fund_data['VL_QUOTA'].dropna()
        daily_returns = quota_series.pct_change().dropna()
        monthly_quota = quota_series.resample('M').last()
        monthly_returns = monthly_quota.pct_change().dropna()
        
        # Helper functions
        def calc_return(start_date, end_date):
            period_data = quota_series[(quota_series.index >= start_date) & 
                                    (quota_series.index <= end_date)]
            # Check if we have data at or near the start date (within 7 days tolerance)
            if len(period_data) >= 2:
                actual_start = period_data.index[0]
                date_diff = (actual_start - start_date).days
                # Accept if first observation is within 7 days of target start date
                if abs(date_diff) <= 7:
                    return (period_data.iloc[-1] / period_data.iloc[0] - 1)
            return np.nan
        
        def calc_benchmark_return(start_date, end_date):
            if cdi is None:
                return np.nan
            period_cdi = cdi[(cdi.index >= start_date) & (cdi.index <= end_date)]
            if len(period_cdi) > 0:
                return (1 + period_cdi).prod() - 1
            return np.nan
        
        current_date = latest_date
        current_month_start = pd.Timestamp(current_date.year, current_date.month, 1)
        current_year_start = pd.Timestamp(current_date.year, 1, 1)
        
        # Calculate period returns
        result['RETURN_MTD'] = calc_return(current_month_start, current_date)
        result['RETURN_3M'] = calc_return(current_date - pd.DateOffset(months=3), current_date)
        result['RETURN_6M'] = calc_return(current_date - pd.DateOffset(months=6), current_date)
        result['RETURN_YTD'] = calc_return(current_year_start, current_date)
        result['RETURN_12M'] = calc_return(current_date - pd.DateOffset(months=12), current_date)
        result['RETURN_24M'] = calc_return(current_date - pd.DateOffset(months=24), current_date)
        result['RETURN_36M'] = calc_return(current_date - pd.DateOffset(months=36), current_date)
        result['RETURN_TOTAL'] = (1 + daily_returns).prod() - 1
        
        result['BCHMK_3M'] = calc_benchmark_return(current_date - pd.DateOffset(months=3), current_date)
        result['BCHMK_6M'] = calc_benchmark_return(current_date - pd.DateOffset(months=6), current_date)
        result['BCHMK_12M'] = calc_benchmark_return(current_date - pd.DateOffset(months=12), current_date)
        result['BCHMK_24M'] = calc_benchmark_return(current_date - pd.DateOffset(months=24), current_date)
        result['BCHMK_36M'] = calc_benchmark_return(current_date - pd.DateOffset(months=36), current_date)
        
        # Calculate BCHMK_TOTAL aligned to fund's actual date range
        if cdi is not None and len(daily_returns) > 0:
            fund_start_date = daily_returns.index[0]
            fund_end_date = daily_returns.index[-1]
            result['BCHMK_TOTAL'] = calc_benchmark_return(fund_start_date, fund_end_date)
        else:
            result['BCHMK_TOTAL'] = np.nan
        
        for period in ['3M', '6M', '12M', '24M', '36M', 'TOTAL']:
            fund_ret = result[f'RETURN_{period}']
            bench_ret = result[f'BCHMK_{period}']
            if not np.isnan(fund_ret) and not np.isnan(bench_ret) and bench_ret != 0:
                result[f'EXCESS_{period}'] = (fund_ret / bench_ret)
            else:
                result[f'EXCESS_{period}'] = np.nan
        
        # Monthly metrics
        if len(monthly_returns) > 0:
            total_months = len(monthly_returns)
            months_above_zero = (monthly_returns > 0).sum()
            result['M_ABOVE_0'] = (months_above_zero / total_months) if total_months > 0 else np.nan
            
            if cdi is not None:
                cdi_monthly = (1 + cdi).resample('M').prod() - 1
                aligned_data = pd.DataFrame({'fund': monthly_returns, 'cdi': cdi_monthly}).dropna()
                
                if len(aligned_data) > 0:
                    months_above_benchmark = (aligned_data['fund'] > aligned_data['cdi']).sum()
                    result['M_ABOVE_BCHMK'] = (months_above_benchmark / len(aligned_data))
                else:
                    result['M_ABOVE_BCHMK'] = np.nan
            else:
                result['M_ABOVE_BCHMK'] = np.nan
            
            result['BEST_MONTH'] = monthly_returns.max()
            result['WORST_MONTH'] = monthly_returns.min()
        else:
            result['M_ABOVE_0'] = np.nan
            result['M_ABOVE_BCHMK'] = np.nan
            result['BEST_MONTH'] = np.nan
            result['WORST_MONTH'] = np.nan
        
        # Volatility metrics
        def calc_volatility(start_date):
            period_returns = daily_returns[daily_returns.index >= start_date]
            # Check if we have data at or near the start date (within 7 days tolerance)
            if len(period_returns) >= 2:
                actual_start = period_returns.index[0]
                date_diff = (actual_start - start_date).days
                # Accept if first observation is within 7 days of target start date
                if abs(date_diff) <= 7:
                    return period_returns.std() * np.sqrt(252)
            return np.nan
        
        result['VOL_12M'] = calc_volatility(current_date - pd.DateOffset(months=12))
        result['VOL_24M'] = calc_volatility(current_date - pd.DateOffset(months=24))
        result['VOL_36M'] = calc_volatility(current_date - pd.DateOffset(months=36))
        result['VOL_TOTAL'] = daily_returns.std() * np.sqrt(252) if len(daily_returns) >= 2 else np.nan
        
        # Sharpe ratios
        for period in ['12M', '24M', '36M', 'TOTAL']:
            ret_key = f'RETURN_{period}'
            vol_key = f'VOL_{period}'
            if not np.isnan(result[ret_key]) and not np.isnan(result[vol_key]) and result[vol_key] != 0:
                days = {'12M': 240, '24M': 480, '36M': 720, 'TOTAL': len(daily_returns)}[period]
                ann_return = (1 + result[ret_key]) ** (252 / days) - 1
                result[f'SHARPE_{period}'] = ann_return / result[vol_key]
            else:
                result[f'SHARPE_{period}'] = np.nan
        
        # Risk metrics
        if len(daily_returns) > 0:
            gains_threshold = np.percentile(daily_returns, 95)
            cvar_gains = daily_returns[daily_returns >= gains_threshold].mean()
            losses_threshold = np.percentile(daily_returns, 5)
            cvar_losses = daily_returns[daily_returns <= losses_threshold].mean()
            
            if not np.isnan(cvar_gains) and not np.isnan(cvar_losses) and cvar_losses != 0:
                result['RACHEV_DAILY'] = cvar_gains / abs(cvar_losses) if cvar_gains > 0 else np.nan
            else:
                result['RACHEV_DAILY'] = np.nan
            
            gains = daily_returns[daily_returns > 0].sum()
            losses = abs(daily_returns[daily_returns < 0].sum())
            result['OMEGA_DAILY'] = gains / losses if losses != 0 else np.nan
            
            var_95_daily = np.percentile(daily_returns, 5)
            result['VAR_95_D'] = var_95_daily
            result['CVAR_95_D'] = daily_returns[daily_returns <= var_95_daily].mean()
        
        # Monthly risk metrics
        if len(monthly_returns) > 0:
            gains_threshold_monthly = np.percentile(monthly_returns, 95)
            cvar_gains_monthly = monthly_returns[monthly_returns >= gains_threshold_monthly].mean()
            losses_threshold_monthly = np.percentile(monthly_returns, 5)
            cvar_losses_monthly = monthly_returns[monthly_returns <= losses_threshold_monthly].mean()
            
            if not np.isnan(cvar_gains_monthly) and not np.isnan(cvar_losses_monthly) and cvar_losses_monthly != 0:
                result['RACHEV_MONTHLY'] = cvar_gains_monthly / abs(cvar_losses_monthly) if cvar_gains_monthly > 0 else np.nan
            else:
                result['RACHEV_MONTHLY'] = np.nan

            monthly_gains = monthly_returns[monthly_returns > 0].sum()
            monthly_losses = abs(monthly_returns[monthly_returns < 0].sum())
            result['OMEGA_MONTHLY'] = monthly_gains / monthly_losses if monthly_losses != 0 else np.nan
            
            var_95_monthly = np.percentile(monthly_returns, 5)
            result['VAR_95_M'] = var_95_monthly
            result['CVAR_95_M'] = monthly_returns[monthly_returns <= var_95_monthly].mean()

        # Weekly risk metrics
        weekly_quota = quota_series.resample('W').last()
        weekly_returns = weekly_quota.pct_change().dropna()

        if len(weekly_returns) > 0:
            # Weekly Rachev Ratio
            gains_threshold_weekly = np.percentile(weekly_returns, 95)
            cvar_gains_weekly = weekly_returns[weekly_returns >= gains_threshold_weekly].mean()
            losses_threshold_weekly = np.percentile(weekly_returns, 5)
            cvar_losses_weekly = weekly_returns[weekly_returns <= losses_threshold_weekly].mean()
            
            if not np.isnan(cvar_gains_weekly) and not np.isnan(cvar_losses_weekly) and cvar_losses_weekly != 0:
                result['RACHEV_WEEKLY'] = cvar_gains_weekly / abs(cvar_losses_weekly) if cvar_gains_weekly > 0 else np.nan
            else:
                result['RACHEV_WEEKLY'] = np.nan

            # Weekly Omega Ratio
            weekly_gains = weekly_returns[weekly_returns > 0].sum()
            weekly_losses = abs(weekly_returns[weekly_returns < 0].sum())
            result['OMEGA_WEEKLY'] = weekly_gains / weekly_losses if weekly_losses != 0 else np.nan
            
            # Weekly VaR and CVaR (95%)
            var_95_weekly = np.percentile(weekly_returns, 5)
            result['VAR_95_W'] = var_95_weekly
            result['CVAR_95_W'] = weekly_returns[weekly_returns <= var_95_weekly].mean()
        else:
            result['RACHEV_WEEKLY'] = np.nan
            result['OMEGA_WEEKLY'] = np.nan
            result['VAR_95_W'] = np.nan
            result['CVAR_95_W'] = np.nan
        
        # Drawdown metrics
        if len(quota_series) >= 2:
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            # Maximum Drawdown (MDD)
            result['MDD'] = drawdown.min()
            
            # Conditional Drawdown at Risk (CDaR) - 95%
            alpha = 0.05  # 95% confidence level
            sorted_drawdowns = np.sort(drawdown.values)
            threshold_idx = int(len(sorted_drawdowns) * alpha)
            
            if threshold_idx > 0:
                worst_drawdowns = sorted_drawdowns[:threshold_idx]
                result['CDAR_95'] = worst_drawdowns.mean()
            else:
                # If we have very few observations, use the worst drawdown
                result['CDAR_95'] = sorted_drawdowns[0] if len(sorted_drawdowns) > 0 else np.nan
            
            # MDD Duration (days underwater)
            is_underwater = drawdown < 0
            underwater_periods = []
            current_period = 0
            
            for underwater in is_underwater:
                if underwater:
                    current_period += 1
                else:
                    if current_period > 0:
                        underwater_periods.append(current_period)
                    current_period = 0
            
            if current_period > 0:
                underwater_periods.append(current_period)
            
            result['MDD_DAYS'] = max(underwater_periods) if underwater_periods else 0
        else:
            result['MDD'] = np.nan
            result['CDAR_95'] = np.nan
            result['MDD_DAYS'] = np.nan
        
        results.append(result)
    
    output_df = pd.DataFrame(results)
    
    # Merge with fund info
    if info_df is not None:
        # Get columns from Fund_Guide - now including Tributação and Liquidez
        info_columns_to_keep = ['Fundo de Investimento', 'CNPJ', 'Status', 'Gestor', 
                                'Categoria BTG', 'Subcategoria BTG', 'Tributação', 'Liquidez', 'Suitability']
        
        # Only keep columns that exist in info_df
        available_columns = [col for col in info_columns_to_keep if col in info_df.columns]
        info_df_subset = info_df[available_columns].copy()
        
        output_df = info_df_subset.merge(output_df, left_on='CNPJ', right_on='CNPJ_FUNDO', how='left')
        output_df = output_df.drop(columns=['CNPJ_FUNDO'])
        
        metric_columns = [col for col in output_df.columns if col not in available_columns]
        output_df = output_df[available_columns + metric_columns]
    
    output_df.columns = output_df.columns.str.upper()
    output_df['FUNDO DE INVESTIMENTO'] = output_df['FUNDO DE INVESTIMENTO'].str.upper()
    output_df = output_df.fillna('n/a')
    
    return output_df


def calculate_stock_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive risk and performance metrics for each stock.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with datetime index and columns representing different stocks
        
    Returns:
    --------
    pd.DataFrame
        Metrics for each stock as rows
    """
    
    metrics = {}
    current_year = pd.Timestamp.now().year
    
    for stock in prices.columns:
        # Get clean price data for this specific stock only
        stock_prices = prices[stock].dropna()
        
        if len(stock_prices) == 0:
            continue
        
        # Calculate returns at different frequencies from stock's own data
        stock_returns_d = stock_prices.pct_change().dropna()
        
        # Weekly returns: resample to week-end, then calculate returns
        stock_prices_w = stock_prices.resample('W').last().dropna()
        stock_returns_w = stock_prices_w.pct_change().dropna()
        
        # Monthly returns: resample to month-end, then calculate returns
        stock_prices_m = stock_prices.resample('M').last().dropna()
        stock_returns_m = stock_prices_m.pct_change().dropna()
        
        # Skip if insufficient data
        if len(stock_returns_d) == 0:
            continue
            
        metrics_dict = {}
        
        # ===== Risk Metrics (Daily, Weekly, Monthly) =====
        
        def calculate_risk_metrics(returns, suffix):
            """Calculate VaR, CVaR, Omega, and Rachev for given frequency"""
            if len(returns) < 20:  # Minimum threshold for meaningful statistics
                return {
                    f'VaR(95){suffix}': np.nan,
                    f'CVaR(95){suffix}': np.nan,
                    f'Omega{suffix}': np.nan,
                    f'Rachev{suffix}': np.nan
                }
            
            # VaR(95) - Value at Risk at 95% confidence (5th percentile loss)
            var_95 = np.percentile(returns, 5)
            
            # CVaR(95) - Conditional Value at Risk (expected loss beyond VaR)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Omega Ratio - probability-weighted ratio of gains vs losses
            threshold = 0
            returns_above = returns[returns > threshold] - threshold
            returns_below = threshold - returns[returns < threshold]
            omega = returns_above.sum() / returns_below.sum() if returns_below.sum() != 0 else np.nan
            
            # Rachev Ratio - ratio of expected upside to expected downside
            upper_threshold = np.percentile(returns, 95)
            lower_threshold = np.percentile(returns, 5)
            
            upper_cvar = returns[returns >= upper_threshold].mean()
            lower_cvar = returns[returns <= lower_threshold].mean()
            
            # Rachev should be positive (good upside vs bad downside)
            rachev = abs(upper_cvar / lower_cvar) if lower_cvar != 0 else np.nan
            
            return {
                f'VaR(95){suffix}': var_95,
                f'CVaR(95){suffix}': cvar_95,
                f'Omega{suffix}': omega,
                f'Rachev{suffix}': rachev
            }
        
        # Calculate for each frequency
        metrics_dict.update(calculate_risk_metrics(stock_returns_d, '_D'))
        metrics_dict.update(calculate_risk_metrics(stock_returns_w, '_W'))
        metrics_dict.update(calculate_risk_metrics(stock_returns_m, '_M'))
        
        # ===== Drawdown Metrics (Daily only, from prices) =====
        
        # Calculate drawdown from price series (not returns)
        running_max = stock_prices.expanding().max()
        drawdown = (stock_prices - running_max) / running_max
        
        # Maximum Drawdown - worst peak-to-trough decline
        max_drawdown = drawdown.min()
        metrics_dict['Max Drawdown'] = max_drawdown
        
        # Conditional Drawdown - average of worst 5% drawdowns (CVaR of drawdowns)
        n_worst = max(1, int(len(drawdown) * 0.05))
        worst_drawdowns = drawdown.nsmallest(n_worst)
        conditional_drawdown = worst_drawdowns.mean()
        metrics_dict['Conditional Drawdown'] = conditional_drawdown
        
        # ===== Accumulated Returns =====
        
        current_date = stock_prices.index[-1]
        
        def get_period_return(prices, months):
            """Calculate return for a given period in months"""
            try:
                start_date = current_date - pd.DateOffset(months=months)
                mask = prices.index >= start_date
                
                if mask.sum() == 0:
                    return np.nan
                    
                period_prices = prices[mask]
                
                # Need at least 2 points to calculate return
                if len(period_prices) < 2:
                    return np.nan
                
                # Calculate total return over period
                return (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
            except Exception as e:
                return np.nan
        
        # Multi-period returns (trailing returns)
        for months in [12, 24, 36, 48, 60]:
            metrics_dict[f'Return {months}M'] = get_period_return(stock_prices, months)

        # Annual returns (calendar year returns)
        for year in range(2019, 2027):
            try:
                # Skip incomplete years (current year and future)
                if year >= current_year:
                    metrics_dict[f'Return {year}'] = np.nan
                    continue
                
                # Get data for this specific year
                year_data = stock_prices[stock_prices.index.year == year]
                
                # Only calculate if stock has meaningful data for that year
                # Require at least 30 trading days (~2 months of data)
                if len(year_data) >= 30:
                    year_return = (year_data.iloc[-1] / year_data.iloc[0]) - 1
                    metrics_dict[f'Return {year}'] = year_return
                else:
                    metrics_dict[f'Return {year}'] = np.nan
                    
            except Exception as e:
                metrics_dict[f'Return {year}'] = np.nan
        
        # Store metrics for this stock
        metrics[stock] = metrics_dict
    
    # Convert to DataFrame
    result_df = pd.DataFrame(metrics).T
    
    # Reorder columns for better readability
    column_order = [
        # Risk metrics - Daily
        'VaR(95)_D', 'CVaR(95)_D', 'Omega_D', 'Rachev_D',
        # Risk metrics - Weekly
        'VaR(95)_W', 'CVaR(95)_W', 'Omega_W', 'Rachev_W',
        # Risk metrics - Monthly
        'VaR(95)_M', 'CVaR(95)_M', 'Omega_M', 'Rachev_M',
        # Drawdown metrics
        'Max Drawdown', 'Conditional Drawdown',
        # Returns
        'Return 12M', 'Return 24M', 'Return 36M', 'Return 48M', 'Return 60M',
        'Return 2025', 'Return 2024', 'Return 2023', 
        'Return 2022', 'Return 2021', 'Return 2020', 'Return 2019',
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in result_df.columns]
    result_df = result_df[column_order]
    
    return result_df

