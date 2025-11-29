"""
Football Data Scraper for GitHub Actions
----------------------------------------
Fetches latest data from FBref using soccerdata and saves as CSV.
Leagues: TUR, ENG, ESP, ITA, GER, FRA
"""

import soccerdata as sd
import pandas as pd
import os
import sys
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

# Fetch current and previous season to ensure we have enough history for models
SEASONS = ['2023', '2024']
DATA_DIR = 'data'

def main():
    print("🚀 Starting Football Data Scraper...")
    
    # Create data directory if not exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"   Created directory: {DATA_DIR}")
        
    for league_code, prefix in LEAGUES.items():
        print(f"\n⚽ Processing: {league_code} ({prefix})")
        
        try:
            # Initialize FBref scraper
            # no_cache=True forces fresh download (important for GitHub Actions)
            # no_store=False saves raw htmls locally (optional, maybe disable to save space)
            fb = sd.FBref(leagues=league_code, seasons=SEASONS)
            
            # 1. Fixture & Results
            print("   > Downloading Schedule...")
            schedule = fb.read_schedule()
            # Flatten MultiIndex if present
            if isinstance(schedule.columns, pd.MultiIndex):
                schedule.columns = ['_'.join(col).strip() for col in schedule.columns.values]
            
            csv_path = os.path.join(DATA_DIR, f"{prefix}_fikstur.csv")
            schedule.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # 2. Standard Stats
            print("   > Downloading Standard Stats...")
            stats = fb.read_team_season_stats(stat_type="standard")
            if isinstance(stats.columns, pd.MultiIndex):
                stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
                
            csv_path = os.path.join(DATA_DIR, f"{prefix}_stats.csv")
            stats.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
            # 3. Shooting Stats
            print("   > Downloading Shooting Stats...")
            shooting = fb.read_team_season_stats(stat_type="shooting")
            if isinstance(shooting.columns, pd.MultiIndex):
                shooting.columns = ['_'.join(col).strip() for col in shooting.columns.values]
                
            csv_path = os.path.join(DATA_DIR, f"{prefix}_shooting.csv")
            shooting.to_csv(csv_path)
            print(f"     Saved: {csv_path}")
            
        except Exception as e:
            print(f"❌ Error processing {league_code}: {e}")
            # Don't stop the script, try next league
            continue
            
    print("\n✅ Scraping Completed!")

if __name__ == "__main__":
    main()
