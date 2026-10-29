import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
from datetime import datetime

# Настройка интерфейса страницы
st.set_page_config(page_title="Deribit Crypto GEX", layout="wide")
st.title("📊 Крипто GEX Дашборд (Deribit API)")
st.caption("Точный расчет Gamma Exposure (GEX) маркетмейкеров по формуле Блэка-Шоулза")

# Справочная функция расчета Гаммы по Блэку-Шоулза
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# 1. Выбор доступных активов в боковой панели
assets = ["BTC", "ETH", "SOL", "USDC"]
selected_asset = st.sidebar.selectbox("Выберите актив:", assets, index=0)

# Загрузка опционных инструментов вынесена выше, чтобы использовать их для симуляции цены при сбое сети
@st.cache_data(ttl=300)
def get_instruments(currency):
    url = f"https://deribit.com{currency}&kind=option&expired=false"
    try:
        response = requests.get(url, timeout=5).json()
        df = pd.DataFrame(response['result'])
        df['expiration_date'] = df['instrument_name'].apply(lambda x: x.split('-')[1])
        return df
    except Exception:
        return pd.DataFrame()

instruments_df = get_instruments(selected_asset)

# 2. Многоуровневое получение текущей цены (Spot)
@st.cache_data(ttl=30)
def get_spot_price(currency, df_options):
    # Попытка 1: Deribit
    url_deribit = f"https://deribit.com{currency.lower()}_usd"
    try:
        response = requests.get(url_deribit, timeout=4).json()
        if 'result' in response and 'index_price' in response['result']:
            return float(response['result']['index_price']), "Deribit API"
    except Exception:
        pass

    # Попытка 2: Binance
    symbol_binance = f"{currency.upper()}USDT"
    url_binance = f"https://binance.com{symbol_binance}"
    try:
        response = requests.get(url_binance, timeout=4).json()
        if 'price' in response:
            return float(response['price']), "Binance API"
    except Exception:
        pass

    # Попытка 3: Автоопределение по центральному страйку (Если сеть полностью лежит)
    if not df_options.empty:
        try:
            # Извлекаем все страйки из имен инструментов
            strikes = df_options['instrument_name'].apply(lambda x: float(x.split('-')[2])).unique()
            if len(strikes) > 0:
                median_strike = float(np.median(strikes))
                return median_strike, "Автоопределение (ATM Strike)"
        except Exception:
            pass

    # Дефолтные константы на самый крайний случай
    defaults = {"BTC": 68000.0, "ETH": 2500.0, "SOL": 150.0, "USDC": 1.0}
    return defaults.get(currency.upper(), 100.0), "Резервный пресет"

if instruments_df.empty:
    st.error("Не удалось загрузить цепочку опционов с Deribit из-за ограничений сети. Попробуйте обновить страницу (Rerun) через несколько секунд.")
else:
    # Определяем цену
    spot_price, price_source = get_spot_price(selected_asset, instruments_df)
    st.sidebar.info(f"Источник цены: {price_source}")

    # Сортируем даты экспирации
    try:
        expirations = sorted(instruments_df['expiration_date'].unique(), key=lambda x: datetime.strptime(x, '%d%b%y'))
    except Exception:
        expirations = sorted(instruments_df['expiration_date'].unique())
        
    selected_exp = st.sidebar.selectbox("Выберите дату экспирации:", expirations, index=0)
    filtered_instruments = instruments_df[instruments_df['expiration_date'] == selected_exp]['instrument_name'].tolist()

    # 4. Загрузка книги заявок и расчет GEX
    @st.cache_data(ttl=60)
    def get_gex_data(instrument_list, S, currency):
        gex_records = []
        url = f"https://deribit.com{currency}&kind=option"
        try:
            response = requests.get(url, timeout=7).json()
            summary_df = pd.DataFrame(response['result'])
            summary_df = summary_df[summary_df['instrument_name'].isin(instrument_list)]
        except Exception:
            return pd.DataFrame()
        
        for _, row in summary_df.iterrows():
            name = row['instrument_name']
            parts = name.split('-')
            if len(parts) < 4:
                continue
            
            strike = float(parts[2])
            option_type = parts[3]
            
            oi = float(row.get('open_interest', 0))
            iv = float(row.get('bid_iv', 0) + row.get('ask_iv', 0)) / 200.0
            
            if iv == 0:
                iv = float(row.get('mark_iv', 0)) / 100.0
                
            exp_str = parts[1]
            try:
                exp_date = datetime.strptime(exp_str, '%d%b%y')
                time_to_exp = (exp_date - datetime.utcnow()).total_seconds() / (365 * 24 * 3600)
            except Exception:
                continue
            
            if time_to_exp <= 0 or iv <= 0 or oi == 0:
                continue
                
            gamma = calculate_gamma(S, strike, time_to_exp, r=0.04, sigma=iv)
            gex_val = gamma * oi * (S ** 2) * 0.01
            
            if option_type == 'P':
                gex_val = -gex_val
                
            gex_records.append({
                'strike': strike,
                'type': 'Call' if option_type == 'C' else 'Put',
                'gex': gex_val,
                'oi': oi
            })
            
        return pd.DataFrame(gex_records)

    with st.spinner("Загрузка биржевых стаканов и расчет GEX..."):
        gex_df = get_gex_data(filtered_instruments, spot_price, selected_asset)

    if gex_df.empty:
        st.warning("Биржа временно ограничила частые запросы данных (Rate Limit). Пожалуйста, подождите 10-15 секунд и нажмите кнопку Rerun в правом верхнем углу меню.")
    else:
        net_gex = gex_df.groupby('strike')['gex'].sum().reset_index()
        
        min_strike = spot_price * 0.7
        max_strike = spot_price * 1.3
        net_gex_filtered = net_gex[(net_gex['strike'] >= min_strike) & (net_gex['strike'] <= max_strike)].sort_values('strike')

        if net_gex_filtered.empty:
            net_gex_filtered = net_gex.sort_values('strike')

        call_wall = net_gex_filtered.loc[net_gex_filtered['gex'].idxmax()]['strike']
        put_wall = net_gex_filtered.loc[net_gex_filtered['gex'].idxmin()]['strike']
        
        gamma_flip = spot_price
        for i in range(len(net_gex_filtered)-1):
            if (net_gex_filtered.iloc[i]['gex'] < 0 and net_gex_filtered.iloc[i+1]['gex'] > 0) or \
               (net_gex_filtered.iloc[i]['gex'] > 0 and net_gex_filtered.iloc[i+1]['gex'] < 0):
                gamma_flip = (net_gex_filtered.iloc[i]['strike'] + net_gex_filtered.iloc[i+1]['strike']) / 2
                break

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Цена {selected_asset}", f"${spot_price:,.2f}")
        col2.metric("Call Wall", f"${call_wall:,.0f}")
        col3.metric("Put Wall", f"${put_wall:,.0f}")
        col4.metric("Точка Gamma Flip", f"${gamma_flip:,.0f}")

        st.markdown("---")

        fig = px.bar(
            net_gex_filtered,
            x='strike',
            y='gex',
            title=f"Профиль Net Gamma Exposure (GEX) на {selected_exp} для {selected_asset}",
            labels={'strike': 'Страйк опциона ($)', 'gex': 'Dollar GEX ($ на 1% движения)'},
            color='gex',
            color_continuous_scale=px.colors.diverging.RdYlGn
        )
        
        fig.add_vline(x=spot_price, line_width=3, line_dash="dash", line_color="blue", annotation_text="Spot")
        st.plotly_chart(fig, use_container_width=True)
