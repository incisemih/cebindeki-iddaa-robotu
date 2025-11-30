"""
Football Data Scraper for GitHub Actions
----------------------------------------
Fetches latest data (Season 2025) from FBref using soccerdata.
Includes rate limiting and error handling.
"""

import soccerdata as sd
import pandas as pd
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
LEAGUES = {
    "TUR-Super Lig": "TUR",
    "ENG-Premier League": "ENG",
    "ESP-La Liga": "ESP",
    "ITA-Serie A": "ITA",
    "GER-Bundesliga": "GER",
    "FRA-Ligue 1": "FRA"
}

# Fetching Season 2025 (2025-2026)
SEASONS = ['2025']
DATA_DIR = 'data'

def main():
    print("🚀 Starting Football Data Scraper (Season 2025)...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"   Created directory: {DATA_DIR}")

    for league_code, prefix in LEAGUES.items():
        print(f"\n⚽ Processing: {league_code} ({prefix})")
        
        try:
            # Initialize FBref scraper
            fb = sd.FBref(leagues=league_code, seasons=SEASONS)
            
            # 1. Fixture & Results
            print(f"   > Downloading Schedule...")
            schedule = fb.read_schedule()
            
            if isinstance(schedule.columns, pd.MultiIndex):
                schedule.columns = ['_'.join(col).strip() for col in schedule.columns.values]
            
            csv_path = os.path.join(DATA_DIR, f"{prefix}_fikstur.csv")
            schedule.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # 2. Standard Stats
            print(f"   > Downloading Standard Stats...")
            stats = fb.read_team_season_stats(stat_type="standard")
            if isinstance(stats.columns, pd.MultiIndex):
                stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
                
            csv_path = os.path.join(DATA_DIR, f"{prefix}_stats.csv")
            stats.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # 3. Shooting Stats
            print(f"   > Downloading Shooting Stats...")
            shooting = fb.read_team_season_stats(stat_type="shooting")
            if isinstance(shooting.columns, pd.MultiIndex):
                shooting.columns = ['_'.join(col).strip() for col in shooting.columns.values]
                
            csv_path = os.path.join(DATA_DIR, f"{prefix}_shooting.csv")
            shooting.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # Rate Limiting
            print("   ⏳ Waiting 4 seconds (Rate Limit Protection)...")
            time.sleep(4)
            
        except Exception as e:
            print(f"❌ ERROR: Could not process {league_code}.")
            print(f"   Reason: {e}")
            print("   (Skipping to next league...)")
            continue
            
    print("\n✅ Scraping Session Completed!")

if __name__ == "__main__":
    main()
