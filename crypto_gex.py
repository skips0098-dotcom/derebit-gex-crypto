import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
from datetime import datetime

# Настройка интерфейса
st.set_page_config(page_title="Deribit Crypto GEX", layout="wide")
st.title("📊 Крипто GEX Дашборд (Deribit API)")
st.caption("Точный расчет Gamma Exposure (GEX) маркетмейкеров по формуле Блэка-Шоулза")

# Справочная функция расчета Гаммы по Блэку-Шоулзу
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# 1. Получение доступных валют
assets = ["BTC", "ETH", "SOL", "USDC"]
selected_asset = st.sidebar.selectbox("Выберите актив:", assets, index=0)

# 2. Получение текущей цены (Spot) с Deribit
@st.cache_data(ttl=30)
def get_spot_price(currency):
    url = f"https://deribit.com{currency.lower()}_usd"
    try:
        response = requests.get(url).json()
        return response['result']['index_price']
    except Exception:
        return None

spot_price = get_spot_price(selected_asset)

if not spot_price:
    st.error("Не удалось получить текущую цену с Deribit.")
else:
    # 3. Получение всех активных инструментов (опционов)
    @st.cache_data(ttl=300)
    def get_instruments(currency):
        url = f"https://deribit.com{currency}&kind=option&expired=false"
        response = requests.get(url).json()
        df = pd.DataFrame(response['result'])
        # Извлекаем дату экспирации из названия (например, BTC-27MAR26-60000-C)
        df['expiration_date'] = df['instrument_name'].apply(lambda x: x.split('-')[1])
        return df

    instruments_df = get_instruments(selected_asset)
    expirations = sorted(instruments_df['expiration_date'].unique(), key=lambda x: datetime.strptime(x, '%d%b%y'))
    
    selected_exp = st.sidebar.selectbox("Выберите дату экспирации:", expirations, index=0)

    # Фильтруем инструменты по выбранной дате
    filtered_instruments = instruments_df[instruments_df['expiration_date'] == selected_exp]['instrument_name'].tolist()

    # 4. Получение среза рынка (Orderbook / Book Summary) по выбранным опционам
    @st.cache_data(ttl=60)
    def get_gex_data(instrument_list, S):
        gex_records = []
        
        # Запрашиваем данные пачками (или через срез), для скорости используем book_summary
        url = f"https://deribit.com{selected_asset}&kind=option"
        response = requests.get(url).json()
        
        summary_df = pd.DataFrame(response['result'])
        summary_df = summary_df[summary_df['instrument_name'].isin(instrument_list)]
        
        for _, row in summary_df.iterrows():
            name = row['instrument_name']
            parts = name.split('-')
            strike = float(parts[2])
            option_type = parts[3] # 'C' или 'P'
            
            oi = float(row.get('open_interest', 0))
            iv = float(row.get('bid_iv', 0) + row.get('ask_iv', 0)) / 200.0 # перевод из процентов в доли (0.50 вместо 50)
            
            if iv == 0:
                iv = float(row.get('mark_iv', 0)) / 100.0
                
            # Расчет времени до экспирации T (в годах)
            exp_str = parts[1]
            exp_date = datetime.strptime(exp_str, '%d%b%y')
            time_to_exp = (exp_date - datetime.utcnow()).total_seconds() / (365 * 24 * 3600)
            
            if time_to_exp <= 0 or iv <= 0 or oi == 0:
                continue
                
            # Считаем точную Гамму по Блэку-Шоулзу
            gamma = calculate_gamma(S, strike, time_to_exp, r=0.04, sigma=iv)
            
            # Расчет классического GEX (с учетом размера контракта на Deribit: для BTC/ETH/SOL размер равен 1 монете)
            # Формула: Gamma * Open Interest * Spot^2 (или Spot для приведения к фиату)
            # Для крипты принято считать Dollar GEX = Gamma * OI * S^2 * 0.01
            gex_val = gamma * oi * (S ** 2) * 0.01
            
            if option_type == 'P':
                gex_val = -gex_val # Путы шортят ММ
                
            gex_records.append({
                'strike': strike,
                'type': 'Call' if option_type == 'C' else 'Put',
                'gex': gex_val,
                'oi': oi
            })
            
        return pd.DataFrame(gex_records)

    with st.spinner("Загрузка и расчет GEX..."):
        gex_df = get_gex_data(filtered_instruments, spot_price)

    if gex_df.empty:
        st.warning("Нет достаточных данных (IV или OI равны нулю) для расчета на эту дату.")
    else:
        # Агрегируем коллы и путы по страйкам
        net_gex = gex_df.groupby('strike')['gex'].sum().reset_index()
        
        # Фильтруем страйки вокруг текущей цены для наглядности (-30% / +30%)
        min_strike = spot_price * 0.7
        max_strike = spot_price * 1.3
        net_gex_filtered = net_gex[(net_gex['strike'] >= min_strike) & (net_gex['strike'] <= max_strike)].sort_values('strike')

        # Ключевые точки
        call_wall = net_gex_filtered.loc[net_gex_filtered['gex'].idxmax()]['strike']
        put_wall = net_gex_filtered.loc[net_gex_filtered['gex'].idxmin()]['strike']
        
        # Расчет Gamma Flip (смена знака)
        gamma_flip = spot_price
        for i in range(len(net_gex_filtered)-1):
            if (net_gex_filtered.iloc[i]['gex'] < 0 and net_gex_filtered.iloc[i+1]['gex'] > 0) or \
               (net_gex_filtered.iloc[i]['gex'] > 0 and net_gex_filtered.iloc[i+1]['gex'] < 0):
                gamma_flip = (net_gex_filtered.iloc[i]['strike'] + net_gex_filtered.iloc[i+1]['strike']) / 2
                break

        # Сетка метрик
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Цена {selected_asset} (Spot)", f"${spot_price:,.2f}")
        col2.metric("Call Wall (Абс. максимум GEX)", f"${call_wall:,.0f}")
        col3.metric("Put Wall (Абс. минимум GEX)", f"${put_wall:,.0f}")
        col4.metric("Точка Gamma Flip", f"${gamma_flip:,.0f}")

        st.markdown("---")

        # График
        fig = px.bar(
            net_gex_filtered,
            x='strike',
            y='gex',
            title=f"Профиль Net Gamma Exposure (GEX) на {selected_exp} для {selected_asset}",
            labels={'strike': 'Страйк ($)', 'gex': 'Dollar GEX ($ на 1% движения)'},
            color='gex',
            color_continuous_scale=px.colors.diverging.RdYlGn
        )
        
        # Добавляем вертикальную линию текущей цены
        fig.add_vline(x=spot_price, line_width=3, line_dash="dash", line_color="blue", annotation_text="Spot")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Посмотреть сырые данные"):
            st.dataframe(net_gex_filtered)
