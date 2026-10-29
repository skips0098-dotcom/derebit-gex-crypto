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

# Справочная функция расчета Гаммы по Блэку-Шоулзу
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# 1. Выбор доступных активов в боковой панели
assets = ["BTC", "ETH", "SOL", "USDC"]
selected_asset = st.sidebar.selectbox("Выберите актив:", assets, index=0)

# 2. Многоуровневое получение текущей цены (Spot) для стабильности в облаке
@st.cache_data(ttl=30)
def get_spot_price(currency):
    # Попытка 1: Напрямую с Deribit
    url_deribit = f"https://deribit.com{currency.lower()}_usd"
    try:
        response = requests.get(url_deribit, timeout=5).json()
        if 'result' in response and 'index_price' in response['result']:
            return float(response['result']['index_price'])
    except Exception:
        pass

    # Попытка 2: Запасной вариант через Binance API (если IP облака временно заблокирован Cloudflare)
    symbol_binance = f"{currency.upper()}USDT"
    url_binance = f"https://binance.com{symbol_binance}"
    try:
        response = requests.get(url_binance, timeout=5).json()
        if 'price' in response:
            return float(response['price'])
    except Exception:
        pass

    # Попытка 3: Резерв через CoinGecko
    cg_mapping = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "USDC": "usd-coin"}
    cg_id = cg_mapping.get(currency.upper(), "bitcoin")
    url_coingecko = f"https://coingecko.com{cg_id}&vs_currencies=usd"
    try:
        response = requests.get(url_coingecko, timeout=5).json()
        return float(response[cg_id]['usd'])
    except Exception:
        return None

spot_price = get_spot_price(selected_asset)

if not spot_price:
    st.error("Не удалось получить текущую цену ни из одного источника (Deribit, Binance, CoinGecko). Проверьте интернет-соединение.")
else:
    # 3. Получение всех активных инструментов (опционов) с биржи
    @st.cache_data(ttl=300)
    def get_instruments(currency):
        url = f"https://deribit.com{currency}&kind=option&expired=false"
        try:
            response = requests.get(url, timeout=10).json()
            df = pd.DataFrame(response['result'])
            # Извлекаем дату экспирации из названия (например, BTC-27MAR26-60000-C)
            df['expiration_date'] = df['instrument_name'].apply(lambda x: x.split('-')[1])
            return df
        except Exception:
            return pd.DataFrame()

    instruments_df = get_instruments(selected_asset)

    if instruments_df.empty:
        st.error("Не удалось загрузить цепочку опционов с Deribit. Возможно, API биржи временно недоступно.")
    else:
        # Сортируем даты экспирации по хронологии
        try:
            expirations = sorted(instruments_df['expiration_date'].unique(), key=lambda x: datetime.strptime(x, '%d%b%y'))
        except Exception:
            expirations = sorted(instruments_df['expiration_date'].unique())
            
        selected_exp = st.sidebar.selectbox("Выберите дату экспирации:", expirations, index=0)

        # Фильтруем доступные опционы под выбранную дату
        filtered_instruments = instruments_df[instruments_df['expiration_date'] == selected_exp]['instrument_name'].tolist()

        # 4. Загрузка стакана/среза рынка и расчет GEX
        @st.cache_data(ttl=60)
        def get_gex_data(instrument_list, S, currency):
            gex_records = []
            url = f"https://deribit.com{currency}&kind=option"
            try:
                response = requests.get(url, timeout=10).json()
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
                option_type = parts[3] # 'C' (Call) или 'P' (Put)
                
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
                    
                # Расчет Гаммы по Блэку-Шоулзу
                gamma = calculate_gamma(S, strike, time_to_exp, r=0.04, sigma=iv)
                
                # Расчет Dollar GEX (объем фиата на 1% изменения базового актива)
                gex_val = gamma * oi * (S ** 2) * 0.01
                
                # Путы имеют противоположную для маркетмейкера экспозицию
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
            st.warning("Нет достаточных рыночных данных (открытого интереса или волатильности) для построения профиля на эту дату.")
        else:
            # Агрегируем чистый GEX по каждому страйку
            net_gex = gex_df.groupby('strike')['gex'].sum().reset_index()
            
            # Обрезаем график вокруг спота (-30% / +30%), чтобы убрать пустые далекие страйки
            min_strike = spot_price * 0.7
            max_strike = spot_price * 1.3
            net_gex_filtered = net_gex[(net_gex['strike'] >= min_strike) & (net_gex['strike'] <= max_strike)].sort_values('strike')

            if net_gex_filtered.empty:
                net_gex_filtered = net_gex.sort_values('strike')

            # Расчет ключевых опционных уровней
            call_wall = net_gex_filtered.loc[net_gex_filtered['gex'].idxmax()]['strike']
            put_wall = net_gex_filtered.loc[net_gex_filtered['gex'].idxmin()]['strike']
            
            gamma_flip = spot_price
            for i in range(len(net_gex_filtered)-1):
                if (net_gex_filtered.iloc[i]['gex'] < 0 and net_gex_filtered.iloc[i+1]['gex'] > 0) or \
                   (net_gex_filtered.iloc[i]['gex'] > 0 and net_gex_filtered.iloc[i+1]['gex'] < 0):
                    gamma_flip = (net_gex_filtered.iloc[i]['strike'] + net_gex_filtered.iloc[i+1]['strike']) / 2
                    break

            # Вывод метрик на главный экран
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Цена {selected_asset} (Spot)", f"${spot_price:,.2f}")
            col2.metric("Call Wall (Крупное сопротивление)", f"${call_wall:,.0f}")
            col3.metric("Put Wall (Крупная поддержка)", f"${put_wall:,.0f}")
            col4.metric("Точка перехода Gamma Flip", f"${gamma_flip:,.0f}")

            st.markdown("---")

            # Построение графика в Plotly
            fig = px.bar(
                net_gex_filtered,
                x='strike',
                y='gex',
                title=f"Профиль Net Gamma Exposure (GEX) на экспирацию {selected_exp} для {selected_asset}",
                labels={'strike': 'Страйк опциона ($)', 'gex': 'Dollar GEX ($ на 1% движения цены)'},
                color='gex',
                color_continuous_scale=px.colors.diverging.RdYlGn
            )
            
            # Добавление пунктирной линии текущей цены
            fig.add_vline(x=spot_price, line_width=3, line_dash="dash", line_color="blue", annotation_text="Текущая цена (Spot)")
            st.plotly_chart(fig, use_container_width=True)

            # Раскрывающаяся таблица внизу страницы
            with st.expander("Посмотреть детальные данные по страйкам в виде таблицы"):
                st.dataframe(net_gex_filtered.sort_values(by='strike', ascending=True))
