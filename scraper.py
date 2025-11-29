"""
Football Data Scraper for GitHub Actions (Debug Mode)
-----------------------------------------------------
Fetches latest data from FBref using soccerdata and saves as CSV.
Includes logic to list available leagues for debugging names.
"""

import soccerdata as sd
import pandas as pd
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
# We will try these codes. If TUR-Super Lig is wrong, the debug print will help us find the right one.
LEAGUES_TO_TRY = {
    "TUR-Super Lig": "TUR",
    "ENG-Premier League": "ENG",
    "ESP-La Liga": "ESP",
    "ITA-Serie A": "ITA",
    "GER-Bundesliga": "GER",
    "FRA-Ligue 1": "FRA"
}

SEASONS = ['2023', '2024']
DATA_DIR = 'data'

def main():
    print("🚀 Starting Football Data Scraper (Debug Mode)...")
    
    # --- 1. DETECTIVE BLOCK ---
    print("\n🔍 Searching for correct League Names...")
    try:
        # FBref available leagues
        available = sd.FBref.available_leagues() 
        
        print(f"   Found {len(available)} leagues in total.")
        print("   Filtering for 'TUR', 'Turkey', 'Super':")
        
        found_any = False
        for league in available:
            if any(x in league for x in ['TUR', 'Turkey', 'Super']):
                print(f"   -> FOUND: {league}")
                found_any = True
                
        if not found_any:
            print("   -> No matching leagues found with those keywords.")
            
    except Exception as e:
        print(f"   ⚠️ Could not list available leagues: {e}")
        print("   (Proceeding with hardcoded list...)")

    # --- 2. SAFE DOWNLOAD LOOP ---
    print("\n⬇️ Starting Download Loop...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"   Created directory: {DATA_DIR}")

    for league_code, prefix in LEAGUES_TO_TRY.items():
        print(f"\n⚽ Processing: {league_code} ({prefix})")
        
        try:
            # Initialize FBref scraper
            fb = sd.FBref(leagues=league_code, seasons=SEASONS)
            
            # 1. Fixture & Results
            print(f"   > Downloading Schedule for {league_code}...")
            schedule = fb.read_schedule()
            
            if isinstance(schedule.columns, pd.MultiIndex):
                schedule.columns = ['_'.join(col).strip() for col in schedule.columns.values]
            
            csv_path = os.path.join(DATA_DIR, f"{prefix}_fikstur.csv")
            schedule.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # 2. Standard Stats
            print(f"   > Downloading Standard Stats for {league_code}...")
            stats = fb.read_team_season_stats(stat_type="standard")
            if isinstance(stats.columns, pd.MultiIndex):
                stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
                
            csv_path = os.path.join(DATA_DIR, f"{prefix}_stats.csv")
            stats.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # 3. Shooting Stats
            print(f"   > Downloading Shooting Stats for {league_code}...")
            shooting = fb.read_team_season_stats(stat_type="shooting")
            if isinstance(shooting.columns, pd.MultiIndex):
                shooting.columns = ['_'.join(col).strip() for col in shooting.columns.values]
                
            csv_path = os.path.join(DATA_DIR, f"{prefix}_shooting.csv")
            shooting.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
        except Exception as e:
            print(f"❌ ERROR: Could not process {league_code}.")
            print(f"   Reason: {e}")
            print("   (Skipping to next league...)")
            continue
            
    print("\n✅ Scraping Session Completed!")

if __name__ == "__main__":
    main()
