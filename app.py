import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- 1. AYARLAR ---
st.set_page_config(page_title="İddaa Analizör Pro - Semih İNCİ", page_icon="🦅", layout="wide") # Wide layout yaptık

# KENDİ BİLGİLERİNİ GİR
GITHUB_USER = "incisemih"
GITHUB_REPO = "cebindeki-iddaa-robotu"
BRANCH_NAME = "main"

BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH_NAME}/data"

st.title("🦅 AI Scout Pro - İY/MS ve Skor Analizi")
st.markdown("**Motor:** Monte Carlo Simülasyonu (10.000 Maç)")

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

# --- 3. VERİ ÇEKME ---
@st.cache_data(ttl=600)
def verileri_al(kod):
    try:
        fikstur = pd.read_csv(f"{BASE_URL}/{kod}_fikstur.csv")
        # İsim Düzeltme
        fikstur.rename(columns={'home_score': 'HG', 'away_score': 'AG', 'home_team': 'Home', 'away_team': 'Away'}, inplace=True)
        # Skor Parçalama
        if 'score' in fikstur.columns:
            try:
                fikstur[['HG', 'AG']] = fikstur['score'].str.split('–', expand=True)
            except:
                fikstur[['HG', 'AG']] = fikstur['score'].str.split('-', expand=True)
            fikstur['HG'] = pd.to_numeric(fikstur['HG'], errors='coerce')
            fikstur['AG'] = pd.to_numeric(fikstur['AG'], errors='coerce')
        return fikstur
    except:
        return None

df_fikstur = verileri_al(kisa_kod)

# --- 4. MONTE CARLO SİMÜLASYON MOTORU (YENİ!) ---
def monte_carlo_simulasyon(home_xg, away_xg, n_sim=10000):
    # Futbolda gollerin %45'i ilk yarı, %55'i ikinci yarı atılır (Genel İstatistik)
    h_xg_1y = home_xg * 0.45
    a_xg_1y = away_xg * 0.45
    
    h_xg_2y = home_xg * 0.55
    a_xg_2y = away_xg * 0.55
    
    # 10.000 Maçlık Rastgele Skor Üretimi (Poisson Dağılımı ile)
    # İlk Yarı Skorları
    h_goals_1y = np.random.poisson(h_xg_1y, n_sim)
    a_goals_1y = np.random.poisson(a_xg_1y, n_sim)
    
    # İkinci Yarı Skorları
    h_goals_2y = np.random.poisson(h_xg_2y, n_sim)
    a_goals_2y = np.random.poisson(a_xg_2y, n_sim)
    
    # Maç Sonu Skorları
    h_goals_ft = h_goals_1y + h_goals_2y
    a_goals_ft = a_goals_1y + a_goals_2y
    
    # --- İY / MS HESAPLAMA ---
    ht_res = np.where(h_goals_1y > a_goals_1y, '1', np.where(h_goals_1y == a_goals_1y, 'X', '2'))
    ft_res = np.where(h_goals_ft > a_goals_ft, '1', np.where(h_goals_ft == a_goals_ft, 'X', '2'))
    
    ht_ft = np.core.defchararray.add(ht_res, '/')
    ht_ft = np.core.defchararray.add(ht_ft, ft_res)
    
    # Olasılıkları Say
    unique, counts = np.unique(ht_ft, return_counts=True)
    ht_ft_probs = dict(zip(unique, counts / n_sim * 100))
    
    # Skor Olasılıkları
    scores_str = [f"{h}-{a}" for h, a in zip(h_goals_ft, a_goals_ft)]
    unique_s, counts_s = np.unique(scores_str, return_counts=True)
    score_probs = dict(zip(unique_s, counts_s / n_sim * 100))
    
    return ht_ft_probs, score_probs, (h_goals_ft > a_goals_ft).mean(), (h_goals_ft == a_goals_ft).mean(), (h_goals_ft < a_goals_ft).mean()

# --- 5. ARAYÜZ ---
if df_fikstur is not None:
    st.success("✅ Simülasyon Motoru Hazır!")
    
    oynanmis = df_fikstur[df_fikstur['HG'].notna()]
    gelecek = df_fikstur[df_fikstur['HG'].isna()]

    if not oynanmis.empty:
        # Güç Hesaplama
        league_h_avg = oynanmis['HG'].mean()
        league_a_avg = oynanmis['AG'].mean()
        h_att = oynanmis.groupby('Home')['HG'].mean() / league_h_avg
        a_def = oynanmis.groupby('Away')['HG'].mean() / league_h_avg # Deplasmanın yediği
        
        a_att = oynanmis.groupby('Away')['AG'].mean() / league_a_avg
        h_def = oynanmis.groupby('Home')['AG'].mean() / league_a_avg # Evin yediği

        for index, row in gelecek.head(10).iterrows():
            home, away = row['Home'], row['Away']
            
            if home in h_att and away in a_def:
                # xG Hesapla
                h_exp = h_att[home] * a_def[away] * league_h_avg
                a_exp = a_att[away] * h_def[home] * league_away_avg
                
                # SİMÜLASYONU BAŞLAT
                ht_ft_probs, score_probs, p1, p0, p2 = monte_carlo_simulasyon(h_exp, a_exp)
                
                # KART GÖRÜNÜMÜ
                with st.expander(f"⚽ {home} vs {away}", expanded=True):
                    # 1. Satır: Ana Oranlar
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ev Sahibi (1)", f"%{p1*100:.1f}", delta="Favori" if p1>0.5 else None)
                    c2.metric("Beraberlik (0)", f"%{p0*100:.1f}")
                    c3.metric("Deplasman (2)", f"%{p2*100:.1f}", delta="Favori" if p2>0.5 else None)
                    
                    st.divider()
                    
                    # 2. Satır: İY / MS Analizi (Complicated Part)
                    st.write("🔄 **İY / MS Olasılıkları (Top 3):**")
                    # En yüksek 3 olasılığı bul ve sırala
                    sorted_ht_ft = sorted(ht_ft_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    cols = st.columns(3)
                    for i, (k, v) in enumerate(sorted_ht_ft):
                        cols[i].info(f"**{k}** → %{v:.1f}")
                        
                    st.divider()
                    
                    # 3. Satır: Skor Tahmini
                    st.write("🎯 **Skor Tahmini (Top 3):**")
                    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    scols = st.columns(3)
                    for i, (k, v) in enumerate(sorted_scores):
                        scols[i].success(f"Skor: **{k}** (%{v:.1f})")

    else:
        st.warning("Veri Yetersiz.")
else:
    st.error("GitHub verisi okunamadı.")

