# %% 
# ==========================================
# 1. IMPORTS & CONFIGURATION
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import warnings
warnings.filterwarnings('ignore')

# %% 
# ==========================================
# 2. UNIVERS COMPLET (SANS CLASSIFICATION)
# ==========================================
all_tickers = list(set([
    # Original Core
    'BYDDY', 'HYMLF', 'KPCPY', '39IB:LN', 'SSNHZ', 'XIACF', 'TCTZF', 'TSM', 
    'HXSCF', 'PLBL', 'PDD', 'MPNGY', 'SKM', 'JD', 'LNVGF', 'NTES', 'HIMX', 
    'OWLS', 'APWC', 'YDES', 'PMRTY', 'PASW', 'CYATY', 'TCOM', 'CICHY', 
    'BACHY', 'KB', 'PKX', 'WF', 'KT', 'DDI', 'FGCO', 'MX', 'HDB', 'IBN', 
    'INFY', 'WIT', 'MMYT', 'SIFY', 'BKKLY', 'MFI', 'MSGY', 
    'SIMO', 'PLUT', 'PSIG', 'UCL', 'CHOW', 'AAGIY',
    'LEGN', 'PIAIF', 'YUMC', 'TAL', 'ATAT', 'ATHM', 'RERE', 'API', 'SUGP', 'EDU','TME','HDB','DBSDF','BHP',
    # Original Satellite
    'PUTRY', 'INDO', 'CSPCY', 'BNCM', 'BDUUY', 'WUXAY', 'BABA', 'MENS', 
    'AGCC', 'PERF', 'GGR', 'CPNG', 'HNHPF', 'ACGBY', 'IDCBY', 'KEP', 
    'RDY', 'YTRA', 'TBVPF', 'LUD', 'OCG', 'NHTC', 'HCM', 'TWG', 'GRGC', 
    'ANPA', 'BUUU', 'MSC', 'ZIJMY', 'CMCLF'
]))

# Mapping des Devises
currency_map = {
    '.KS': 'KRW=X', '.HK': 'HKD=X', '.NS': 'INR=X', 
    '.TW': 'TWD=X', '.KL': 'MYR=X', '.SA': 'BRL=X'
}

# %% 
# ==========================================
# 3. CLASSIFICATION AUTOMATIQUE CORE/SATELLITE
# ==========================================
def classify_core_satellite_smart(tickers, 
                                   threshold_mcap=3e9,        # $3 Mds
                                   threshold_volume_usd=15e6,  # $15M/jour
                                   min_history_days=1260):     # 5 ans
    """
    Classification automatique basée sur :
    - Market Cap
    - Liquidité (volume en USD)
    - Historique
    - Risque de delisting / OTC
    """
    core = []
    satellite = []
    failed = []
    
    print("\n" + "="*70)
    print("🤖 CLASSIFICATION AUTOMATIQUE CORE / SATELLITE")
    print("="*70)
    print(f"Critères CORE : Cap>${threshold_mcap/1e9:.0f}Mds | Vol>${threshold_volume_usd/1e6:.0f}M$ | 5+ ans")
    print("-"*70)
    
    for i, ticker in enumerate(tickers, 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="max")
            
            # Extraction des données
            mcap = info.get('marketCap', 0)
            avg_volume = info.get('averageVolume', 0)
            current_price = info.get('regularMarketPrice', info.get('previousClose', 1))
            volume_usd = avg_volume * current_price
            history_days = len(hist)
            exchange = info.get('exchange', 'UNKNOWN')
            
            # Détection des red flags
            is_otc = 'OTC' in exchange.upper() or exchange == 'UNKNOWN'
            is_penny = current_price < 1
            is_delisting_risk = 'delisting' in info.get('longBusinessSummary', '').lower()
            
            # Décision de classification
            meets_mcap = mcap > threshold_mcap
            meets_volume = volume_usd > threshold_volume_usd
            meets_history = history_days > min_history_days
            no_red_flags = not (is_otc or is_penny or is_delisting_risk)
            
            # Classification
            if meets_mcap and meets_volume and meets_history and no_red_flags:
                core.append(ticker)
                category = "✅ CORE"
                reason = f"Cap:{mcap/1e9:.1f}B Vol:{volume_usd/1e6:.0f}M"
            else:
                satellite.append(ticker)
                category = "🛰️ SAT "
                reasons = []
                if not meets_mcap: reasons.append(f"Cap:{mcap/1e9:.1f}B")
                if not meets_volume: reasons.append(f"Vol:{volume_usd/1e6:.0f}M")
                if not meets_history: reasons.append(f"Hist:{history_days}j")
                if not no_red_flags: reasons.append("⚠️RedFlag")
                reason = " | ".join(reasons)
            
            print(f"{i:3d}. {ticker:10s} → {category} | {reason}")
            
        except Exception as e:
            satellite.append(ticker)
            failed.append(ticker)
            print(f"{i:3d}. {ticker:10s} → 🛰️ SAT  | ⚠️ Erreur téléchargement")
    
    print("-"*70)
    print(f"📊 RÉSULTAT : {len(core)} CORE | {len(satellite)} SATELLITE")
    print(f"⚠️  Échecs téléchargement : {len(failed)} titres")
    print("="*70 + "\n")
    
    return core, satellite, failed

# Lancement de la classification
print("🚀 Analyse de l'univers complet...")
core_tickers, satellite_tickers, failed_tickers = classify_core_satellite_smart(all_tickers)

# Sauvegarde pour référence
classification_df = pd.DataFrame({
    'Ticker': core_tickers + satellite_tickers,
    'Category': ['Core']*len(core_tickers) + ['Satellite']*len(satellite_tickers)
})

# %% 
# ==========================================
# 4. TÉLÉCHARGEMENT & CONVERSION FX
# ==========================================
def fetch_and_convert_to_usd(ticker_list, start_date="2018-01-01"):
    data_frames = []
    success_count = 0
    print(f"--- Traitement du lot : {len(ticker_list)} titres ---")
    
    for ticker in ticker_list:
        try:
            stock = yf.download(ticker, start=start_date, progress=False)['Close']
            if stock.empty: continue

            fx_ticker = None
            for suffix, currency in currency_map.items():
                if ticker.endswith(suffix): fx_ticker = currency
            
            if fx_ticker:
                try:
                    fx = yf.download(fx_ticker, start=start_date, progress=False)['Close']
                    aligned = pd.concat([stock, fx], axis=1).dropna()
                    aligned.columns = ['Price', 'FX']
                    converted = aligned['Price'] / aligned['FX']
                    converted.name = ticker
                    data_frames.append(converted)
                    success_count += 1
                except:
                    stock.name = ticker
                    data_frames.append(stock)
                    success_count += 1
            else:
                stock.name = ticker
                data_frames.append(stock)
                success_count += 1
        except: pass     
        
    print(f"Téléchargés avec succès : {success_count}/{len(ticker_list)}")
    return pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()

print("\nRécupération Core...")
core_data = fetch_and_convert_to_usd(core_tickers)
print("Récupération Satellite...")
satellite_data = fetch_and_convert_to_usd(satellite_tickers)

# Fusion et Nettoyage
raw_assets = pd.concat([core_data, satellite_data], axis=1)
min_history = 750
valid_counts = raw_assets.count()
to_keep = valid_counts[valid_counts >= min_history].index
all_assets = raw_assets[to_keep].fillna(method='ffill').dropna()

# ✅ Mise à jour de la classification après nettoyage
core_tickers = [t for t in core_tickers if t in all_assets.columns]
satellite_tickers = [t for t in satellite_tickers if t in all_assets.columns]

print(f"\n📊 Univers Final : {all_assets.shape[1]} actifs | {all_assets.shape[0]} jours")
print(f"   • Core: {len(core_tickers)} titres ({len(core_tickers)/all_assets.shape[1]:.1%})")
print(f"   • Satellite: {len(satellite_tickers)} titres ({len(satellite_tickers)/all_assets.shape[1]:.1%})")

# %% 
# ==========================================
# 5. LISTES SECTORIELLES (POUR CONTRAINTES)
# ==========================================
china_tickers = [
    'BABA', 'PDD', 'JD', 'NTES', 'TCOM', 'BIDU', 'XIACF', 'TCTZF',
    'CICHY', 'BACHY', 'ACGBY', 'IDCBY', 'SKYC',
    'ZIJMY', 'CMCLF', 'OCG', 'UCL', 'ANPA', 'CHOW', 'BUUU', 'MSC', 'CSPCY',
    'PIAIF', 'YUMC', 'TAL', 'ATAT', 'ATHM', 'RERE', 'API', 'SUGP', 'EDU'
]

financial_tickers = [
    'KPCPY', 'CICHY', 'BACHY', 'KB', 'ACGBY', 'IDCBY', 'WF', 
    'HDB', 'IBN', 'BKKLY', 'PLUT', 'AGCC', 'YDES', 'PIAIF'
]

tech_tickers = [
    'TSM', 'SSNHZ', 'HXSCF', 'XIACF', 'TCTZF', '39IB:LN', 
    'BABA', 'PDD', 'JD', 'NTES', 'HIMX', 'SIMO', 'MX', 'PLBL', 
    'INFY', 'WIT', 'SIFY', 'UCL', 'ATHM', 'API', 'RERE'
]

biotech_tickers = ['LEGN']

# %% 
# ==========================================
# 6. MOTEUR ROLLING WINDOW
# ==========================================
lookback = 252
rebalance_freq = 'Q'
rebalance_dates = all_assets.resample(rebalance_freq).last().index
history_weights = {}
history_returns = []

print(f"\n🚀 Lancement backtest avec contraintes dynamiques...")

for i, date in enumerate(rebalance_dates):
    if date < all_assets.index[lookback]: continue
    
    train_data = all_assets.loc[:date].tail(lookback)
    train_data = train_data.loc[:, train_data.var() > 0.000001]
    
    try:
        mu = expected_returns.mean_historical_return(train_data)
        S = risk_models.CovarianceShrinkage(train_data).ledoit_wolf()
        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')
        
        ef = EfficientFrontier(mu, S)
        
        # Classification dynamique Core/Sat
        curr_assets = train_data.columns.tolist()
        curr_core = [t for t in core_tickers if t in curr_assets]
        curr_sat = [t for t in satellite_tickers if t in curr_assets]
        mapper = {}
        for t in curr_core: mapper[t] = "Core"
        for t in curr_sat: mapper[t] = "Satellite"
        
        # Contraintes
        if len(curr_core) > 0 and len(curr_sat) > 0:
            ef.add_sector_constraints(mapper, {"Core": 0.70, "Satellite": 0.0}, {"Core": 1.0, "Satellite": 0.30})
        
        curr_china = [t for t in china_tickers if t in curr_assets]
        if curr_china:
            china_indices = [ef.tickers.index(t) for t in curr_china]
            ef.add_constraint(lambda w: sum(w[i] for i in china_indices) <= 0.35)
            
        curr_fin = [t for t in financial_tickers if t in curr_assets]
        if curr_fin:
            fin_indices = [ef.tickers.index(t) for t in curr_fin]
            ef.add_constraint(lambda w: sum(w[i] for i in fin_indices) <= 0.25)

        curr_tech = [t for t in tech_tickers if t in curr_assets]
        if curr_tech:
            tech_indices = [ef.tickers.index(t) for t in curr_tech]
            ef.add_constraint(lambda w: sum(w[i] for i in tech_indices) >= 0.20)

        if 'SKYC' in curr_assets:
            idx = ef.tickers.index('SKYC')
            ef.add_constraint(lambda w: w[idx] <= 0.02)
            
        if '3661.TW' in curr_assets:
            idx = ef.tickers.index('3661.TW')
            ef.add_constraint(lambda w: w[idx] <= 0.05)
            
        # ✅ CONTRAINTE RENFORCÉE : Max 8% par ligne
        ef.add_constraint(lambda w: w <= 0.08)
        
        weights = ef.max_sharpe(risk_free_rate=0.04)
        cleaned = ef.clean_weights(cutoff=0.02)
        cleaned = {k: v / sum(cleaned.values()) for k, v in cleaned.items()}
        history_weights[date] = cleaned
        
    except:
        try:
            ef = EfficientFrontier(mu, S)
            if 'SKYC' in curr_assets:
                idx = ef.tickers.index('SKYC')
                ef.add_constraint(lambda w: w[idx] <= 0.03)
            # ✅ Mode relaxation : max 8% par ligne aussi
            ef.add_constraint(lambda w: w <= 0.08)
            weights = ef.max_sharpe(risk_free_rate=0.04)
            cleaned = ef.clean_weights(cutoff=0.02)
            history_weights[date] = cleaned
        except:
            if history_weights: 
                cleaned = history_weights[list(history_weights.keys())[-1]]
                history_weights[date] = cleaned
            else: continue 

    # Application
    if i < len(rebalance_dates) - 1:
        next_date = rebalance_dates[i+1]
        period_ret = all_assets.loc[date:next_date].iloc[1:].pct_change().dropna()
    else:
        period_ret = all_assets.loc[date:].iloc[1:].pct_change().dropna()
        
    if not period_ret.empty:
        w_s = pd.Series(cleaned)
        common = period_ret.columns.intersection(w_s.index)
        if len(common) > 0:
            port_ret = (period_ret[common] * w_s[common]).sum(axis=1)
            history_returns.append(port_ret)

# %% 
# ==========================================
# 7. RÉSULTATS & KPIs
# ==========================================
if not history_returns:
    print("❌ ERREUR : Aucun rendement calculé.")
else:
    full_returns = pd.concat(history_returns)
    portfolio_nav = (1 + full_returns).cumprod() * 100

    n_years = len(portfolio_nav) / 252
    total_return = (portfolio_nav.iloc[-1] / 100) - 1
    cagr = (portfolio_nav.iloc[-1] / 100) ** (1 / n_years) - 1
    volatility = full_returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / volatility
    rolling_max = portfolio_nav.cummax()
    max_drawdown = ((portfolio_nav - rolling_max) / rolling_max).min()

    print(f"\n{'='*60}")
    print(f"📊 KPIs FINAUX (Classification Auto)")
    print(f"{'='*60}")
    print(f"Rendement Total : {total_return:>8.2%}")
    print(f"CAGR            : {cagr:>8.2%}")
    print(f"Volatilité      : {volatility:>8.2%}")
    print(f"Max Drawdown    : {max_drawdown:>8.2%}")
    print(f"Sharpe Ratio    : {sharpe:>8.2f}")
    print(f"{'='*60}\n")

# %% 
# ==========================================
# 8. BENCHMARK
# ==========================================
if history_returns:
    bench = yf.download("EEM", start=portfolio_nav.index[0], end=portfolio_nav.index[-1], progress=False)['Close']
    bench = bench.reindex(portfolio_nav.index).fillna(method='ffill')
    bench_nav = ((1 + bench.pct_change().fillna(0)).cumprod() * 100).squeeze()
    portfolio_nav = portfolio_nav.squeeze()

    bench_cagr = (bench_nav.iloc[-1] / 100) ** (1 / n_years) - 1
    alpha_annualise = cagr - bench_cagr

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_nav, label=f'Fonds (CAGR {cagr:.1%})', color='#004488', linewidth=2)
    plt.plot(bench_nav, label=f'EEM (CAGR {bench_cagr:.1%})', color='black', linestyle='--', alpha=0.7)
    plt.fill_between(portfolio_nav.index, portfolio_nav, bench_nav, 
                where=(portfolio_nav >= bench_nav), interpolate=True, color='green', alpha=0.1)
    plt.fill_between(portfolio_nav.index, portfolio_nav, bench_nav, 
                where=(portfolio_nav < bench_nav), interpolate=True, color='red', alpha=0.1)
    plt.title(f'Backtest avec Classification Auto | Alpha: {alpha_annualise:.2%}/an', fontweight='bold')
    plt.ylabel('Base 100')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# %% 
# ==========================================
# 9. ALLOCATION ACTUELLE
# ==========================================
if history_weights:
    latest_weights = pd.Series(history_weights[list(history_weights.keys())[-1]])
    latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)

    print("\n" + "="*70)
    print("🎯 ALLOCATION CIBLE AUJOURD'HUI")
    print("="*70)
    
    for i, (ticker, weight) in enumerate(latest_weights.items(), 1):
        categories = []
        if ticker in core_tickers: categories.append("Core")
        if ticker in satellite_tickers: categories.append("Sat")
        if ticker in china_tickers: categories.append("🇨🇳")
        if ticker in financial_tickers: categories.append("🏦")
        if ticker in tech_tickers: categories.append("💻")
        if ticker in biotech_tickers: categories.append("🧬")
        
        cat_str = "|".join(categories) if categories else "?"
        print(f"{i:2d}. {ticker:10s} {weight:6.2%}  [{cat_str}]")
    
    print("-"*70)
    print(f"Lignes      : {len(latest_weights)}")
    print(f"Core        : {latest_weights[latest_weights.index.isin(core_tickers)].sum():.1%}")
    print(f"Satellite   : {latest_weights[latest_weights.index.isin(satellite_tickers)].sum():.1%}")
    print(f"Chine 🇨🇳    : {latest_weights[latest_weights.index.isin(china_tickers)].sum():.1%}")
    print(f"Finance 🏦  : {latest_weights[latest_weights.index.isin(financial_tickers)].sum():.1%}")
    print(f"Tech 💻     : {latest_weights[latest_weights.index.isin(tech_tickers)].sum():.1%}")
    print("="*70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    latest_weights.plot.barh(ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Pondération')
    ax1.set_title('Allocation par Titre')
    ax1.grid(axis='x', linestyle=':', alpha=0.6)
    
    latest_weights.plot.pie(ax=ax2, autopct='%.1f%%', cmap="tab20")
    ax2.set_title("Répartition")
    ax2.set_ylabel("")
    plt.tight_layout()
    plt.show()

# %%
# ==========================================
# 10. EXPORT DE LA CLASSIFICATION
# ==========================================
print("\n💾 Export de la classification...")
classification_df.to_csv('classification_core_satellite.csv', index=False)
print("✅ Fichier sauvegardé : classification_core_satellite.csv")

# ==========================================
# PRÉPARATION DES DATAFRAMES POUR EXPORT EXCEL
# ==========================================

# Allocation finale
allocation_df = latest_weights.reset_index()
allocation_df.columns = ['Ticker', 'Weight']

allocation_df['Core/Satellite'] = allocation_df['Ticker'].apply(
    lambda x: 'Core' if x in core_tickers else 'Satellite'
)

# Historique des pondérations
weights_df = pd.DataFrame(history_weights).T
weights_df.index.name = 'Date'

# Performance
performance_df = pd.DataFrame({
    'NAV': portfolio_nav,
    'Daily_Return': full_returns
})

# KPIs
kpi_df = pd.DataFrame({
    'Metric': ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Sharpe'],
    'Value': [total_return, cagr, volatility, max_drawdown, sharpe]
})


output_path = r"C:\Users\HP\Documents\M2-222\M2 - 222 - S1\Master's fund\resultats_backtest.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    allocation_df.to_excel(writer, sheet_name='Allocation Finale', index=False)
    weights_df.to_excel(writer, sheet_name='Historique Pondérations')
    performance_df.to_excel(writer, sheet_name='Performance')
    kpi_df.to_excel(writer, sheet_name='KPIs', index=False)
    classification_df.to_excel(writer, sheet_name='Classification', index=False)

print("✅ EXCEL CRÉÉ À :", output_path)
