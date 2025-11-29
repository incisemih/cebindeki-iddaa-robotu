"""
🦅 İddaa Analizör (v1.0)
----------------------------------------------
Author: Antigravity Agent
Description: Mobile-friendly web application for football match prediction using XGBoost.
             Fetches data from GitHub Raw URLs (Cloud Compatible).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

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
# Lütfen kendi GitHub kullanıcı adınızı ve repo isminizi buraya yazın
GITHUB_USER = "incisemih" # Örn: "semih"
GITHUB_REPO = "cebindeki-iddaa-robotu" # Örn: "football-ai"
BRANCH_NAME = "main" # Genelde "main" veya "master"

# --- CONFIGURATION ---
st.set_page_config(
    page_title="İddaa Analizör - Semih İNCİ",
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
    "🇫🇷 Ligue 1 (FRA)": "FRA"
}

# --- CLASSES ---

class GitHubDataFetcher:
    def __init__(self, league_prefix):
        self.league_prefix = league_prefix
        self.base_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH_NAME}/data"
        
    def get_file_url(self, file_type):
        return f"{self.base_url}/{self.league_prefix}_{file_type}.csv"
        
    def fetch_data(self):
        # Construct file URL
        file_url = self.get_file_url("fikstur")
        
        # Debug info for sidebar
        st.sidebar.markdown("### 📡 Bağlantı Durumu")
        st.sidebar.code(file_url, language="text")
        
        try:
            # Read CSV directly from URL
            # storage_options={'User-Agent': 'Mozilla/5.0'} sometimes helps with GitHub
            df = pd.read_csv(file_url, storage_options={'User-Agent': 'Mozilla/5.0'})
            return df
        except Exception as e:
            st.error(f"❌ Veri çekilemedi!")
            st.error(f"🔗 Denenen Adres: {file_url}")
            st.warning("Lütfen kodun başındaki 'GITHUB_USER' ve 'GITHUB_REPO' ayarlarını kontrol edin.")
            st.caption(f"Hata Detayı: {e}")
            return None

class DataProcessor:
    @staticmethod
    def normalize_name(name):
        if not isinstance(name, str): return ""
        name = name.lower().strip()
        replacements = [' sk', ' fk', ' futbol kulübü', 'spor', ' kulübü', ' fc', ' cf']
        for r in replacements: name = name.replace(r, '')
        return name

    def clean_schedule(self, df):
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        cols_map = {
            'date': 'Date',
            'home_team': 'Home',
            'away_team': 'Away',
            'home_score': 'HG',
            'away_score': 'AG',
            'home_xg': 'xG_Home',
            'away_xg': 'xG_Away'
        }
        
        df = df.rename(columns=cols_map)
        
        for col in ['HG', 'AG', 'xG_Home', 'xG_Away']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        conditions = [
            (df['HG'] > df['AG']),
            (df['HG'] < df['AG']),
            (df['HG'] == df['AG'])
        ]
        choices = [1, 2, 0]
        df['Result'] = np.select(conditions, choices, default=np.nan)
        
        if 'xG_Home' in df.columns:
            df['xG_Home'] = df['xG_Home'].fillna(df['HG'])
        else:
            df['xG_Home'] = df['HG']
            
        if 'xG_Away' in df.columns:
            df['xG_Away'] = df['xG_Away'].fillna(df['AG'])
        else:
            df['xG_Away'] = df['AG']
            
        return df

class FeatureEngineering:
    def process(self, df):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')
        
        h_df = df[['Date', 'Home', 'Result', 'xG_Home', 'HG']].rename(columns={'Home': 'Team', 'xG_Home': 'xG', 'HG': 'Goals'})
        h_df['Points'] = h_df['Result'].apply(lambda x: 3 if x==1 else (1 if x==0 else 0))
        h_df['Is_Home'] = 1
        
        a_df = df[['Date', 'Away', 'Result', 'xG_Away', 'AG']].rename(columns={'Away': 'Team', 'xG_Away': 'xG', 'AG': 'Goals'})
        a_df['Points'] = a_df['Result'].apply(lambda x: 3 if x==2 else (1 if x==0 else 0))
        a_df['Is_Home'] = 0
        
        long_df = pd.concat([h_df, a_df]).sort_values(['Team', 'Date'])
        grouped = long_df.groupby('Team')
        
        long_df['Roll_xG_3'] = grouped['xG'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        long_df['Roll_Goals_5'] = grouped['Goals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        long_df['Prev_Date'] = grouped['Date'].shift(1)
        long_df['Fatigue'] = (long_df['Date'] - long_df['Prev_Date']).dt.days.fillna(7)
        
        home_stats = long_df[long_df['Is_Home'] == 1]
        league_home_avg = home_stats['Points'].mean()
        team_home_avg = home_stats.groupby('Team')['Points'].transform(lambda x: x.expanding().mean().shift(1))
        
        long_df['Home_Adv_Factor'] = 0.0
        long_df.loc[long_df['Is_Home'] == 1, 'Home_Adv_Factor'] = team_home_avg - league_home_avg
        long_df['Home_Adv_Factor'] = long_df['Home_Adv_Factor'].fillna(0)
        
        df = df.merge(long_df[['Date', 'Team', 'Roll_xG_3', 'Roll_Goals_5', 'Fatigue', 'Home_Adv_Factor']], 
                      left_on=['Date', 'Home'], right_on=['Date', 'Team'], how='left')
        df = df.rename(columns={'Roll_xG_3': 'H_Roll_xG', 'Roll_Goals_5': 'H_Roll_Goals', 'Fatigue': 'H_Fatigue', 'Home_Adv_Factor': 'H_Adv'}).drop('Team', axis=1)
        
        df = df.merge(long_df[['Date', 'Team', 'Roll_xG_3', 'Roll_Goals_5', 'Fatigue']], 
                      left_on=['Date', 'Away'], right_on=['Date', 'Team'], how='left')
        df = df.rename(columns={'Roll_xG_3': 'A_Roll_xG', 'Roll_Goals_5': 'A_Roll_Goals', 'Fatigue': 'A_Fatigue'}).drop('Team', axis=1)
        
        df = df.fillna(0)
        return df

class XGBoostEngine:
    def __init__(self):
        if MODEL_TYPE == "XGBoost":
            self.model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, eval_metric='mlogloss', use_label_encoder=False)
        else:
            self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
        self.features = ['H_Roll_xG', 'H_Roll_Goals', 'H_Fatigue', 'H_Adv', 'A_Roll_xG', 'A_Roll_Goals', 'A_Fatigue']
        
    def train(self, df):
        train_df = df[df['Date'] < datetime.now()]
        train_df = train_df[train_df['Result'].notnull()] 
        
        if train_df.empty: return 0
        
        X = train_df[self.features]
        y = train_df['Result'].astype(int)
        self.model.fit(X, y)
        return len(train_df)
        
    def predict(self, df):
        now = datetime.now()
        unplayed = df[df['Date'] >= now].sort_values('Date')
        
        if unplayed.empty: return pd.DataFrame()
        
        end_date = unplayed.iloc[0]['Date'] + timedelta(days=3)
        target = unplayed[unplayed['Date'] <= end_date]
        
        if target.empty: return pd.DataFrame()
        
        X = target[self.features]
        probs = self.model.predict_proba(X)
        
        target = target.copy()
        target['p1'] = probs[:, 1] * 100
        target['p0'] = probs[:, 0] * 100
        target['p2'] = probs[:, 2] * 100
        return target

# --- CACHED FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_and_process_data(league_prefix):
    fetcher = GitHubDataFetcher(league_prefix)
    raw_schedule = fetcher.fetch_data()
    
    if raw_schedule is None:
        return None
    
    processor = DataProcessor()
    clean_df = processor.clean_schedule(raw_schedule)
    
    return clean_df

@st.cache_data(ttl=3600)
def run_analysis(df):
    eng = FeatureEngineering()
    processed_df = eng.process(df)
    
    xgb = XGBoostEngine()
    train_size = xgb.train(processed_df)
    preds = xgb.predict(processed_df)
    
    return train_size, preds

# --- MAIN APP ---

def main():
    st.title("🦅 Süper Lig AI Analizör")
    st.caption(f"Powered by {MODEL_TYPE} V15.0 | Cloud Mode")
    
    # Sidebar
    st.sidebar.header("⚙️ Ayarlar")
    
    selected_league_name = st.sidebar.selectbox(
        "Lig Seçiniz",
        list(LEAGUES.keys())
    )
    league_prefix = LEAGUES[selected_league_name]
    
    show_banko = st.sidebar.checkbox("Sadece Banko Maçları Göster", value=False)
    
    if st.sidebar.button("Analizi Başlat", type="primary"):
        with st.spinner(f"{selected_league_name} verileri GitHub'dan çekiliyor..."):
            try:
                # 1. Fetch
                df = fetch_and_process_data(league_prefix)
                
                if df is None:
                    st.stop()
                    
                if df.empty:
                    st.error("Veri dosyası boş.")
                    st.stop()
                    
                # 2. Analyze
                train_size, preds = run_analysis(df)
                
                st.success(f"Veriler yüklendi! {train_size} maç ile model eğitildi.")
                
                # 3. Display
                st.divider()
                st.subheader("📅 Yaklaşan Maçlar (3 Gün)")
                
                if preds.empty:
                    st.info("Yakın tarihte oynanacak maç bulunamadı.")
                else:
                    for _, row in preds.iterrows():
                        max_prob = max(row['p1'], row['p0'], row['p2'])
                        if show_banko and max_prob < 60:
                            continue
                            
                        home = row['Home']
                        away = row['Away']
                        date = row['Date'].strftime("%d.%m %H:%M")
                        
                        p1, p0, p2 = row['p1'], row['p0'], row['p2']
                        
                        rec = ""
                        color = ""
                        comment = ""
                        
                        if p1 > 55:
                            rec = f"MS 1 (%{p1:.1f})"
                            color = "#00FF00"
                            comment = f"{home}, son maçlardaki xG üretimi ({row['H_Roll_xG']:.2f}) ve saha avantajı ile favori."
                        elif p2 > 55:
                            rec = f"MS 2 (%{p2:.1f})"
                            color = "#00FF00"
                            comment = f"{away}, deplasmanda olmasına rağmen form grafiği yüksek. {home} savunmada zorlanabilir."
                        else:
                            rec = f"BERABERE / KG VAR (%{p0:.1f})"
                            color = "#FFA500"
                            comment = f"İki takımın verileri birbirine yakın. Yorgunluk seviyeleri maçı dengeleyebilir."
                        
                        html = f"""
                        <div class="match-card">
                            <div class="match-meta">{date}</div>
                            <div class="match-header">{home} vs {away}</div>
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

    else:
        st.info("👈 Analizi başlatmak için butona tıklayın.")

if __name__ == "__main__":
    main()
