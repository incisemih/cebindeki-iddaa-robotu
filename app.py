"""
🦅 Süper Lig AI Analizör (V30.0) - Streamlit App
----------------------------------------------
Author: Antigravity Agent
Description: Advanced football match prediction using Hybrid Power Ratings (Goals + xG + SoT).
             Fetches "Rich Data" from GitHub Raw URLs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    MODEL_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    MODEL_TYPE = "GradientBoosting"

# Suppress warnings
warnings.filterwarnings('ignore')

# --- GITHUB CONFIGURATION ---
GITHUB_USER = "incisemih" 
GITHUB_REPO = "cebindeki-iddaa-robotu" 
BRANCH_NAME = "main" 

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI HT/FT",
    page_icon="🦅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS FOR MOBILE CARDS ---
st.markdown("""
<style>
    .match-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .match-header {
        font-size: 18px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    .match-meta {
        font-size: 12px;
        color: #AAAAAA;
        margin-bottom: 10px;
    }
    .stats-row {
        display: flex;
        justify-content: space-between;
        background-color: #252525;
        padding: 8px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 13px;
        color: #DDDDDD;
    }
    .prediction-box {
        background-color: #2D2D2D;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        color: #00FF00;
    }
    .comment-box {
        font-size: 13px;
        color: #DDDDDD;
        margin-top: 10px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- LEAGUE MAPPING ---
LEAGUES = {
    "🇹🇷 Süper Lig (TUR)": "TUR",
    "🇬🇧 Premier League (ENG)": "ENG",
    "🇪🇸 La Liga (ESP)": "ESP",
    "🇮🇹 Serie A (ITA)": "ITA",
    "🇩🇪 Bundesliga (GER)": "GER",
    "🇫🇷 Ligue 1 (FRA)": "FRA",
    "🇳🇱 Eredivisie (NED)": "NED",
    "🇵🇹 Liga Portugal (POR)": "POR"
}

# --- CLASSES ---

class GitHubDataFetcher:
    def __init__(self, league_prefix):
        self.league_prefix = league_prefix
        self.base_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH_NAME}/guncel_lig_html"
        
    def get_file_url(self, file_type):
        return f"{self.base_url}/{self.league_prefix}_{file_type}.csv"
        
    def fetch_data(self):
        # 1. Fetch Fixture
        fix_url = self.get_file_url("fikstur")
        # 2. Fetch Team Power (Rich Stats)
        power_url = self.get_file_url("takim_gucu")
        
        st.sidebar.markdown("### 📡 Bağlantı Durumu")
        st.sidebar.code(fix_url, language="text")
        
        try:
            fix_df = pd.read_csv(fix_url, storage_options={'User-Agent': 'Mozilla/5.0'})
            
            try:
                power_df = pd.read_csv(power_url, storage_options={'User-Agent': 'Mozilla/5.0'})
            except:
                st.warning(f"⚠️ '{self.league_prefix}_takim_gucu.csv' bulunamadı. Sadece fikstür verisi kullanılacak.")
                power_df = pd.DataFrame()
                
            return fix_df, power_df
        except Exception as e:
            st.error(f"❌ Veri çekilemedi!")
            st.caption(f"Hata Detayı: {e}")
            return None, None

class DataProcessor:
    @staticmethod
    def calculate_hybrid_power(power_df):
        """
        Calculates Hybrid Attack and Defense Ratings using Rich Stats.
        Formula:
        - Attack: 40% Goals + 40% xG + 20% SoT
        - Defense: 50% Conceded + 50% xGA
        """
        if power_df.empty:
            return {}

        ratings = {}
        
        # Normalize columns if they exist
        # We assume columns like 'Gls', 'xG', 'SoT', 'GA', 'xGA' exist or similar
        # If merged with suffixes, they might be 'Gls', 'xG_standard', etc.
        # Let's try to be flexible or assume standard names from our scraper
        
        # Helper to safely get value
        def get_val(row, col_list):
            for c in col_list:
                if c in row:
                    return float(row[c])
            return 0.0

        for _, row in power_df.iterrows():
            team = row['Squad']
            
            # Attack Components (Per 90 usually, but total is fine if we compare relative)
            # Using totals from the season stats
            gls = get_val(row, ['Gls', 'Goals'])
            xg = get_val(row, ['xG', 'xG_Expected'])
            sot = get_val(row, ['SoT', 'SoT_Shooting'])
            
            # Defense Components
            ga = get_val(row, ['GA', 'GA_Goalkeeping'])
            xga = get_val(row, ['xGA', 'PSxG', 'PSxG_Goalkeeping']) # PSxG is better for GK/Defense eval
            
            # Matches Played (to average)
            mp = get_val(row, ['MP', 'Pl'])
            if mp < 1: mp = 1
            
            # Per Match Averages
            avg_gls = gls / mp
            avg_xg = xg / mp
            avg_sot = sot / mp
            avg_ga = ga / mp
            avg_xga = xga / mp
            
            # Hybrid Formulas
            att_rating = (0.4 * avg_gls) + (0.4 * avg_xg) + (0.2 * (avg_sot / 3)) # SoT scaled down slightly
            def_rating = (0.5 * avg_ga) + (0.5 * avg_xga)
            
            ratings[team] = {
                'att': att_rating,
                'def': def_rating,
                'xg_avg': avg_xg,
                'xga_avg': avg_xga
            }
            
        return ratings

    def clean_schedule(self, df):
        # Basic cleaning similar to V15 but adapted
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        # Rename cols to standard
        cols_map = {'date': 'Date', 'home_team': 'Home', 'away_team': 'Away', 'score': 'Score'}
        df = df.rename(columns=cols_map)
        
        # Parse Score "2–1" -> HG, AG
        if 'Score' in df.columns and 'HG' not in df.columns:
            df[['HG', 'AG']] = df['Score'].str.split('–', expand=True)
        
        for col in ['HG', 'AG']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Result
        conditions = [(df['HG'] > df['AG']), (df['HG'] < df['AG']), (df['HG'] == df['AG'])]
        choices = [1, 2, 0]
        df['Result'] = np.select(conditions, choices, default=np.nan)
        
        return df

class FeatureEngineering:
    def process(self, df, power_ratings):
        df = df.sort_values('Date')
        
        # Add Hybrid Ratings to the dataframe for the model
        # If power_ratings is empty, we use 0 (fallback)
        
        def get_rating(team, kind):
            if team in power_ratings:
                return power_ratings[team].get(kind, 0.0)
            return 0.0

        df['H_Att'] = df['Home'].apply(lambda x: get_rating(x, 'att'))
        df['H_Def'] = df['Home'].apply(lambda x: get_rating(x, 'def'))
        df['A_Att'] = df['Away'].apply(lambda x: get_rating(x, 'att'))
        df['A_Def'] = df['Away'].apply(lambda x: get_rating(x, 'def'))
        
        # Simple Form (Last 5 games points)
        # ... (Simplified for V30 to focus on Hybrid Power, but keeping basic form logic is good)
        # For brevity in this V30 update, we'll rely heavily on the Hybrid Ratings which are season-long robust stats.
        
        return df

class XGBoostEngine:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, eval_metric='mlogloss', use_label_encoder=False)
        self.features = ['H_Att', 'H_Def', 'A_Att', 'A_Def'] # Using Hybrid Ratings as primary features
        
    def train(self, df):
        train_df = df[df['Result'].notnull()]
        if train_df.empty: return 0
        
        X = train_df[self.features]
        y = train_df['Result'].astype(int)
        self.model.fit(X, y)
        return len(train_df)
        
    def predict(self, df):
        now = datetime.now()
        # Filter for upcoming matches (Result is NaN or Date >= Now)
        # Actually, we want matches that haven't been played.
        # In our CSV, unplayed matches usually have NaN Score or are in future.
        
        # Let's assume unplayed matches have NaN Result
        unplayed = df[df['Result'].isnull()].copy()
        
        # Also filter by date to show only next 3-4 days
        unplayed = unplayed[unplayed['Date'] >= now - timedelta(days=1)] # Include today
        unplayed = unplayed.sort_values('Date')
        
        if unplayed.empty: return pd.DataFrame()
        
        # Limit to next 4 days
        end_date = now + timedelta(days=4)
        target = unplayed[unplayed['Date'] <= end_date]
        
        if target.empty: return pd.DataFrame()
        
        X = target[self.features]
        probs = self.model.predict_proba(X)
        
        target['p1'] = probs[:, 1] * 100
        target['p0'] = probs[:, 0] * 100
        target['p2'] = probs[:, 2] * 100
        return target

# --- CACHED FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_and_process_data(league_prefix):
    fetcher = GitHubDataFetcher(league_prefix)
    fix_df, power_df = fetcher.fetch_data()
    
    if fix_df is None: return None, None
    
    processor = DataProcessor()
    clean_fix = processor.clean_schedule(fix_df)
    power_ratings = processor.calculate_hybrid_power(power_df)
    
    return clean_fix, power_ratings

@st.cache_data(ttl=3600)
def run_analysis(df, power_ratings):
    eng = FeatureEngineering()
    processed_df = eng.process(df, power_ratings)
    
    xgb = XGBoostEngine()
    train_size = xgb.train(processed_df)
    preds = xgb.predict(processed_df)
    
    return train_size, preds

# --- MAIN APP ---

def main():
    st.title("🦅 Süper Lig AI Analizör V30")
    st.caption(f"Hibrit Güç Modeli (xG + Şut + Gol) | {MODEL_TYPE}")
    
    # Sidebar
    st.sidebar.header("⚙️ Ayarlar")
    selected_league_name = st.sidebar.selectbox("Lig Seçiniz", list(LEAGUES.keys()))
    league_prefix = LEAGUES[selected_league_name]
    
    if st.sidebar.button("Analizi Başlat", type="primary"):
        with st.spinner(f"{selected_league_name} verileri analiz ediliyor..."):
            try:
                # 1. Fetch
                df, power_ratings = fetch_and_process_data(league_prefix)
                
                if df is None: st.stop()
                if df.empty:
                    st.error("Fikstür dosyası boş.")
                    st.stop()
                    
                # 2. Analyze
                train_size, preds = run_analysis(df, power_ratings)
                
                st.success(f"Veriler yüklendi! {train_size} geçmiş maç ile model eğitildi.")
                
                # 3. Display
                st.divider()
                st.subheader("📅 Yaklaşan Maçlar ve İstatistikler")
                
                if preds.empty:
                    st.info("Yakın tarihte oynanacak maç bulunamadı.")
                else:
                    for _, row in preds.iterrows():
                        home = row['Home']
                        away = row['Away']
                        date = row['Date'].strftime("%d.%m %H:%M")
                        
                        p1, p0, p2 = row['p1'], row['p0'], row['p2']
                        
                        # Get Stats for Display
                        h_stats = power_ratings.get(home, {'xg_avg': 0, 'att': 0})
                        a_stats = power_ratings.get(away, {'xg_avg': 0, 'att': 0})
                        
                        # Recommendation Logic
                        rec = "BERABERE"
                        color = "#FFA500"
                        if p1 > 50:
                            rec = f"MS 1 (%{p1:.0f})"
                            color = "#00FF00"
                        elif p2 > 50:
                            rec = f"MS 2 (%{p2:.0f})"
                            color = "#00FF00"
                            
                        # Comment
                        diff_att = h_stats.get('att',0) - a_stats.get('att',0)
                        if diff_att > 0.5:
                            comment = f"{home}, hücum gücüyle ({h_stats.get('att',0):.2f}) rakibine üstünlük kurabilir."
                        elif diff_att < -0.5:
                            comment = f"{away}, etkili hücum hattıyla ({a_stats.get('att',0):.2f}) deplasmanda tehlikeli."
                        else:
                            comment = "İki takımın güç dengeleri birbirine çok yakın."

                        html = f"""
                        <div class="match-card">
                            <div class="match-meta">{date}</div>
                            <div class="match-header">{home} vs {away}</div>
                            
                            <div class="stats-row">
                                <div>
                                    <span style="color:#888">xG (Ort):</span> <b>{h_stats.get('xg_avg',0):.2f}</b><br>
                                    <span style="color:#888">Güç:</span> <b>{h_stats.get('att',0):.1f}</b>
                                </div>
                                <div style="text-align:right">
                                    <b>{a_stats.get('xg_avg',0):.2f}</b> <span style="color:#888">:(Ort) xG</span><br>
                                    <b>{a_stats.get('att',0):.1f}</b> <span style="color:#888">:Güç</span>
                                </div>
                            </div>
                            
                            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                <div>🏠 {p1:.0f}%</div>
                                <div>🤝 {p0:.0f}%</div>
                                <div>✈️ {p2:.0f}%</div>
                            </div>
                            <div class="prediction-box" style="color: {color}; border: 1px solid {color};">
                                🎯 TAHMİN: {rec}
                            </div>
                            <div class="comment-box">
                                💡 AI: "{comment}"
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.exception(e)

    else:
        st.info("👈 Analizi başlatmak için butona tıklayın.")

if __name__ == "__main__":
    main()
