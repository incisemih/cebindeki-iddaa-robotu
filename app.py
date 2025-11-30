import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- 1. AYARLAR ---
st.set_page_config(page_title="AI Scout Pro - Semih İNCİ", page_icon="🦅", layout="wide")

# KENDİ BİLGİLERİNİ GİR
GITHUB_USER = "incisemih"
GITHUB_REPO = "cebindeki-iddaa-robotu"
BRANCH_NAME = "main"

BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH_NAME}/data"

st.title("🦅 AI Scout Pro - Canlı Takvim")
st.markdown("by Semih İNCİ")

# --- 2. LİG SEÇİMİ ---
lig_secimi = st.selectbox("Ligi Seçin:", [
    "🇬🇧 Premier League (ENG)",
    "🇹🇷 Süper Lig (TUR)",
    "🇪🇸 La Liga (ESP)",
    "🇮🇹 Serie A (ITA)",
    "🇩🇪 Bundesliga (GER)",
    "🇫🇷 Ligue 1 (FRA)"
])

kisa_kod = lig_secimi.split("(")[1].replace(")", "")

# --- 3. VERİ ÇEKME VE ZAMAN DİLİMİ DÜZELTİCİ FONKSİYON ---
@st.cache_data(ttl=600)
def verileri_al(kod):
    try:
        fikstur_url = f"{BASE_URL}/{kod}_fikstur.csv"
        fikstur = pd.read_csv(fikstur_url)

        # İsim Düzeltme (KeyError: 'HG' Çözümü)
        fikstur.rename(columns={
            'home_score': 'HG', 
            'away_score': 'AG', 
            'home_team': 'Home', 
            'away_team': 'Away',
            'date': 'Date',
            'time': 'Time'
        }, inplace=True)
        
        # Skor Parçalama
        if 'score' in fikstur.columns:
            try:
                fikstur[['HG', 'AG']] = fikstur['score'].str.split('–', expand=True)
            except:
                fikstur[['HG', 'AG']] = fikstur['score'].str.split('-', expand=True)
            
            fikstur['HG'] = pd.to_numeric(fikstur['HG'], errors='coerce')
            fikstur['AG'] = pd.to_numeric(fikstur['AG'], errors='coerce')
            
        # --- ZAMAN DİLİMİ VE TARİH DÜZELTMESİ (KeyError: 'DateTime' Çözümü) ---
        fikstur['MatchDateTime'] = pd.to_datetime(fikstur['Date'] + ' ' + fikstur['Time'], errors='coerce')
        
        # UTC'den İstanbul'a çevir (TRT)
        ISTANBUL_TIMEZONE = 'Europe/Istanbul'
        fikstur['MatchDateTime'] = fikstur['MatchDateTime'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        fikstur['MatchDateTime'] = fikstur['MatchDateTime'].dt.tz_convert(ISTANBUL_TIMEZONE)
        
        # Tarih ve saat stringlerini yeniden oluştur
        fikstur['Date_TR'] = fikstur['MatchDateTime'].dt.strftime("%d.%m.%Y")
        fikstur['Time_TR'] = fikstur['MatchDateTime'].dt.strftime("%H:%M")
        
        return fikstur
    except Exception as e:
        return None

df_fikstur = verileri_al(kisa_kod)

# --- 4. SİMÜLASYON MOTORU ---
def dinamik_simulasyon(home_xg, away_xg, n_sim=10000):
    xg_diff = home_xg - away_xg
    
    # Dinamik Ağırlık Hesabı
    if xg_diff > 1.0: split_factor = 0.55 
    elif xg_diff < -1.0: split_factor = 0.55
    elif abs(xg_diff) < 0.3: split_factor = 0.35
    else: split_factor = 0.45 
        
    h_xg_1y = home_xg * split_factor
    a_xg_1y = away_xg * split_factor
    h_xg_2y = home_xg * (1 - split_factor)
    a_xg_2y = away_xg * (1 - split_factor)
    
    # Skor Simülasyonu (SyntaxError çözüldü)
    h_goals_1y = np.random.poisson(h_xg_1y, n_sim)
    a_goals_1y = np.random.poisson(a_xg_1y, n_sim)
    h_goals_2y = np.random.poisson(h_xg_2y, n_sim)
    a_goals_2y = np.random.poisson(a_xg_2y, n_sim)
    
    h_goals_ft = h_goals_1y + h_goals_2y
    a_goals_ft = a_goals_1y + a_goals_2y
    
    ht_res = np.where(h_goals_1y > a_goals_1y, '1', np.where(h_goals_1y == a_goals_1y, 'X', '2'))
    ft_res = np.where(h_goals_ft > a_goals_ft, '1', np.where(h_goals_ft == a_goals_ft, 'X', '2'))
    
    ht_ft = np.char.add(ht_res, '/')
    ht_ft = np.char.add(ht_ft, ft_res)
    
    unique, counts = np.unique(ht_ft, return_counts=True)
    ht_ft_probs = dict(zip(unique, counts / n_sim * 100))
    
    p1 = (h_goals_ft > a_goals_ft).mean()
    p0 = (h_goals_ft == a_goals_ft).mean()
    p2 = (h_goals_ft < a_goals_ft).mean()
    
    return ht_ft_probs, p1, p0, p2

# --- 5. ARAYÜZ ---
if df_fikstur is not None:
    st.success(f"✅ {lig_secimi} Verileri Hazır!")
    
    oynanmis = df_fikstur[df_fikstur['HG'].notna()]
    # HATA ÇÖZÜMÜ: 'DateTime' yerine 'MatchDateTime' ile sıralıyoruz
    gelecek = df_fikstur[df_fikstur['HG'].isna()].sort_values(by='MatchDateTime') 

    if not oynanmis.empty:
        # İstatistikler
        league_h_avg = oynanmis['HG'].mean()
        league_a_avg = oynanmis['AG'].mean()
        h_att = oynanmis.groupby('Home')['HG'].mean() / league_h_avg
        a_def = oynanmis.groupby('Away')['HG'].mean() / league_h_avg
        a_att = oynanmis.groupby('Away')['AG'].mean() / league_a_avg
        h_def = oynanmis.groupby('Home')['AG'].mean() / league_a_avg

        for index, row in gelecek.head(15).iterrows():
            home, away = row['Home'], row['Away']
            mac_tarihi = row['Date_TR']
            mac_saati = row['Time_TR']
            
            if home in h_att and away in a_def:
                h_exp = h_att[home] * a_def[away] * league_h_avg
                a_exp = a_att[away] * h_def[home] * league_a_avg
                
                ht_ft_probs, p1, p0, p2 = dinamik_simulasyon(h_exp, a_exp)
                
                # KART GÖRÜNÜMÜ
                with st.expander(f"📅 {mac_tarihi} ⏰ {mac_saati} | ⚽ {home} vs {away}", expanded=True):
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ev Sahibi", f"%{p1*100:.1f}", delta="Banko" if p1>0.6 else None)
                    c2.metric("Beraberlik", f"%{p0*100:.1f}")
                    c3.metric("Deplasman", f"%{p2*100:.1f}", delta="Banko" if p2>0.5 else None)
                    
                    st.divider()
                    
                    st.write(f"🔄 **İY / MS Senaryoları:**")
                    sorted_ht_ft = sorted(ht_ft_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    col_a, col_b, col_c = st.columns(3)
                    for i, (senaryo, yuzde) in enumerate(sorted_ht_ft):
                        if i == 0: col_a.info(f"🥇 **{senaryo}** (%{yuzde:.1f})")
                        if i == 1: col_b.success(f"🥈 **{senaryo}** (%{yuzde:.1f})")
                        if i == 2: col_c.warning(f"🥉 **{senaryo}** (%{yuzde:.1f})")
            else:
                pass
    else:
        st.warning("Veri Yetersiz.")
else:
    st.error("GitHub verisi okunamadı. Ayarları kontrol edin.")
