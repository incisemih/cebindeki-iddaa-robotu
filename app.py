"""
🦅 Süper Lig AI Analizör (V15) - Streamlit App
----------------------------------------------
Author: Antigravity Agent
Description: Mobile-friendly web application for football match prediction using XGBoost.
"""

import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
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

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Süper Lig AI V15",
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

# --- CLASSES (ADAPTED FOR STREAMLIT) ---

class DataIngestion:
    @staticmethod
    def normalize_name(name):
        if not isinstance(name, str): return ""
        name = name.lower().strip()
        replacements = [' sk', ' fk', ' futbol kulübü', 'spor', ' kulübü']
        for r in replacements: name = name.replace(r, '')
        mapping = {
            'fenerbahce': 'fenerbahçe', 'besiktas': 'beşiktaş', 
            'basaksehir': 'başakşehir', 'caykur rize': 'rizespor',
            'rize': 'rizespor', 'fatih karagümrük': 'karagümrük',
            'gaziantep': 'gaziantep', 'samsun': 'samsunspor',
            'bodrum': 'bodrumspor', 'eyüp': 'eyüpspor',
            'göztepe': 'göztepe'
        }
        return mapping.get(name, name)

    def parse_html_content(self, content):
        try:
            soup = BeautifulSoup(content, 'html.parser')
        except: return pd.DataFrame()
        
        table = None
        for t in soup.find_all('table'):
            if 'sched' in t.get('id', '') or 'fixture' in t.get('id', ''):
                table = t
                break
        
        if not table:
            # Fallback search
            for t in soup.find_all('table'):
                headers = [th.get_text().lower() for th in t.find_all('th')]
                if any('home' in h for h in headers) or any('ev' in h for h in headers):
                    table = t
                    break
                    
        if not table: return pd.DataFrame()
        
        matches = []
        for row in table.find('tbody').find_all('tr'):
            if 'class' in row.attrs and ('thead' in row.attrs['class'] or 'spacer' in row.attrs['class']): continue
            
            h_cell = row.find('td', {'data-stat': 'home_team'})
            a_cell = row.find('td', {'data-stat': 'away_team'})
            s_cell = row.find('td', {'data-stat': 'score'})
            d_cell = row.find('td', {'data-stat': 'date'})
            t_cell = row.find('td', {'data-stat': 'start_time'})
            xg_h_cell = row.find('td', {'data-stat': 'xg_home'})
            xg_a_cell = row.find('td', {'data-stat': 'xg_away'})
            
            if not (h_cell and a_cell): continue
            
            home = self.normalize_name(h_cell.get_text(strip=True))
            away = self.normalize_name(a_cell.get_text(strip=True))
            score = s_cell.get_text(strip=True) if s_cell else ""
            date_str = d_cell.get_text(strip=True) if d_cell else ""
            time_str = t_cell.get_text(strip=True) if t_cell else ""
            
            xg_h = float(xg_h_cell.get_text(strip=True)) if xg_h_cell and xg_h_cell.get_text(strip=True) else None
            xg_a = float(xg_a_cell.get_text(strip=True)) if xg_a_cell and xg_a_cell.get_text(strip=True) else None
            
            hg, ag, res = None, None, None
            if score and re.search(r'\d+[\–\-]\d+', score):
                try:
                    parts = re.split(r'[\–\-]', score)
                    hg, ag = int(parts[0]), int(parts[1])
                    res = 1 if hg > ag else (2 if ag > hg else 0)
                except: pass
            
            if xg_h is None and hg is not None: xg_h = float(hg)
            if xg_a is None and ag is not None: xg_a = float(ag)
                
            matches.append({
                'Date': date_str, 'Time': time_str,
                'Home': home, 'Away': away,
                'HG': hg, 'AG': ag, 'Result': res,
                'xG_Home': xg_h, 'xG_Away': xg_a
            })
            
        return pd.DataFrame(matches)

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
        train_df = df[df['Result'].notnull()]
        train_df = train_df[train_df['Date'] < datetime.now()]
        if train_df.empty: return 0
        
        X = train_df[self.features]
        y = train_df['Result'].astype(int)
        self.model.fit(X, y)
        return len(train_df)
        
    def predict(self, df):
        now = datetime.now()
        unplayed = df[df['Date'] >= now].sort_values('Date')
        if unplayed.empty: return pd.DataFrame()
        
        # Next 7 days
        end_date = unplayed.iloc[0]['Date'] + timedelta(days=7)
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

@st.cache_data
def process_files(uploaded_files):
    ingest = DataIngestion()
    all_data = pd.DataFrame()
    
    for file in uploaded_files:
        content = file.getvalue().decode("utf-8")
        df = ingest.parse_html_content(content)
        if not df.empty:
            all_data = pd.concat([all_data, df])
            
    if not all_data.empty:
        all_data = all_data.drop_duplicates(subset=['Date', 'Home', 'Away'])
        
    return all_data

@st.cache_data
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
    st.caption(f"Powered by {MODEL_TYPE} V15.0 | Self-Improving Architecture")
    
    # Sidebar
    st.sidebar.header("📂 Veri Yükleme")
    st.sidebar.info("Lütfen güncel HTML dosyalarını yükleyin.")
    
    uploaded_files = st.sidebar.file_uploader(
        "Fikstür ve İstatistik Dosyaları", 
        type=['html'], 
        accept_multiple_files=True
    )
    
    show_banko = st.sidebar.checkbox("Sadece Banko Maçları Göster", value=False)
    
    if not uploaded_files:
        st.warning("👈 Analize başlamak için lütfen sol menüden dosya yükleyin.")
        st.stop()
        
    # Process
    with st.spinner("Veriler işleniyor ve Model eğitiliyor..."):
        try:
            raw_df = process_files(uploaded_files)
            
            if raw_df.empty:
                st.error("Yüklenen dosyalardan maç verisi okunamadı. Lütfen doğru dosyaları yüklediğinizden emin olun.")
                st.stop()
                
            train_size, preds = run_analysis(raw_df)
            
            st.success(f"Model {train_size} maç ile eğitildi ve hazır! 🚀")
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
            st.stop()
            
    # Display Results
    st.divider()
    st.subheader("📅 Yaklaşan Maçlar")
    
    if preds.empty:
        st.info("Yakın tarihte oynanacak maç bulunamadı.")
    else:
        for _, row in preds.iterrows():
            # Filter Banko
            max_prob = max(row['p1'], row['p0'], row['p2'])
            if show_banko and max_prob < 60:
                continue
                
            # Prepare Data
            home = row['Home'].title()
            away = row['Away'].title()
            date = row['Date'].strftime("%d.%m %H:%M")
            
            p1, p0, p2 = row['p1'], row['p0'], row['p2']
            
            # Logic
            rec = ""
            color = ""
            comment = ""
            
            if p1 > 55:
                rec = f"MS 1 (%{p1:.1f})"
                color = "#00FF00" # Green
                comment = f"{home}, son maçlardaki xG üretimi ({row['H_Roll_xG']:.2f}) ve saha avantajı ile favori."
            elif p2 > 55:
                rec = f"MS 2 (%{p2:.1f})"
                color = "#00FF00"
                comment = f"{away}, deplasmanda olmasına rağmen form grafiği yüksek. {home} savunmada zorlanabilir."
            else:
                rec = f"BERABERE / KG VAR (%{p0:.1f})"
                color = "#FFA500" # Orange
                comment = f"İki takımın verileri birbirine yakın. Yorgunluk seviyeleri maçı dengeleyebilir."
            
            # Card HTML
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

if __name__ == "__main__":
    main()
