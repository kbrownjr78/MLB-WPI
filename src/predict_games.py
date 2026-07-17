import os
import asyncio
import datetime
import numpy as np
import pandas as pd
import statsapi
import pybaseball as pb
import python_weather

# Enable pybaseball caching to optimize resource utilization
pb.cache.enable()

class MLBFullPropQuantEngine:
    def __init__(self):
        self.today = datetime.date.today().strftime('%Y-%m-%d')
        self.current_year = datetime.date.today().year
        
        # 1. Official MLB Team-to-City Mapping (for Live Weather Queries)
        self.team_cities = {
            'Arizona Diamondbacks': 'Phoenix', 'Atlanta Braves': 'Atlanta', 'Baltimore Orioles': 'Baltimore',
            'Boston Red Sox': 'Boston', 'Chicago Cubs': 'Chicago', 'Chicago White Sox': 'Chicago',
            'Cincinnati Reds': 'Cincinnati', 'Cleveland Guardians': 'Cleveland', 'Colorado Rockies': 'Denver',
            'Detroit Tigers': 'Detroit', 'Houston Astros': 'Houston', 'Kansas City Royals': 'Kansas City',
            'Los Angeles Angels': 'Anaheim', 'Los Angeles Dodgers': 'Los Angeles', 'Miami Marlins': 'Miami',
            'Milwaukee Brewers': 'Milwaukee', 'Minnesota Twins': 'Minneapolis', 'New York Mets': 'New York',
            'New York Yankees': 'New York', 'Oakland Athletics': 'Oakland', 'Philadelphia Phillies': 'Philadelphia',
            'Pittsburgh Pirates': 'Pittsburgh', 'San Diego Padres': 'San Diego', 'San Francisco Giants': 'San Francisco',
            'Seattle Mariners': 'Seattle', 'St. Louis Cardinals': 'St. Louis', 'Tampa Bay Rays': 'St. Petersburg',
            'Texas Rangers': 'Arlington', 'Toronto Blue Jays': 'Toronto', 'Washington Nationals': 'Washington'
        }
        
        # 2. FanGraphs-to-MLB Team Name Normalization Map
        self.fg_team_map = {
            'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
            'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
            'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
            'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
            'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
            'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
            'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN'
        }
        
        # 3. Authentic Base Park Factors & Stadium Dome Registry
        self.park_data = {
            'Arizona Diamondbacks': {'factor': 1.02, 'dome': True},   'Atlanta Braves': {'factor': 0.99, 'dome': False},
            'Baltimore Orioles': {'factor': 1.01, 'dome': False},     'Boston Red Sox': {'factor': 1.08, 'dome': False},
            'Chicago Cubs': {'factor': 1.00, 'dome': False},          'Chicago White Sox': {'factor': 1.03, 'dome': False},
            'Cincinnati Reds': {'factor': 1.12, 'dome': False},       'Cleveland Guardians': {'factor': 0.98, 'dome': False},
            'Colorado Rockies': {'factor': 1.32, 'dome': False},      'Detroit Tigers': {'factor': 0.97, 'dome': False},
            'Houston Astros': {'factor': 0.98, 'dome': True},         'Kansas City Royals': {'factor': 1.02, 'dome': False},
            'Los Angeles Angels': {'factor': 0.98, 'dome': False},    'Los Angeles Dodgers': {'factor': 0.97, 'dome': False},
            'Miami Marlins': {'factor': 0.95, 'dome': True},          'Milwaukee Brewers': {'factor': 1.01, 'dome': True},
            'Minnesota Twins': {'factor': 0.99, 'dome': False},       'New York Mets': {'factor': 0.95, 'dome': False},
            'New York Yankees': {'factor': 1.04, 'dome': False},      'Oakland Athletics': {'factor': 0.93, 'dome': False},
            'Philadelphia Phillies': {'factor': 1.02, 'dome': False}, 'Pittsburgh Pirates': {'factor': 0.97, 'dome': False},
            'San Diego Padres': {'factor': 0.94, 'dome': False},      'San Francisco Giants': {'factor': 0.93, 'dome': False},
            'Seattle Mariners': {'factor': 0.92, 'dome': True},       'St. Louis Cardinals': {'factor': 0.97, 'dome': False},
            'Tampa Bay Rays': {'factor': 0.96, 'dome': True},         'Texas Rangers': {'factor': 1.02, 'dome': True},
            'Toronto Blue Jays': {'factor': 1.00, 'dome': True},      'Washington Nationals': {'factor': 1.00, 'dome': False}
        }
    async def get_live_weather(self, city):
        """Asynchronously pulls real-time weather parameters using python-weather."""
        try:
            async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
                weather = await client.get(city)
                raw_dir = weather.wind_direction
                dir_val = raw_dir.value if hasattr(raw_dir, 'value') else str(raw_dir)
                return {
                    'temp': weather.temperature,
                    'wind_speed': weather.wind_speed,
                    'wind_direction': dir_val
                }
        except Exception as e:
            print(f"Weather fetch failed for {city}: {e}. Applying standard default baselines.")
            return {'temp': 70, 'wind_speed': 0, 'wind_direction': 0}

    def fetch_live_schedule(self):
        """Ingests current day games directly from MLB StatsAPI."""
        try:
            raw_games = statsapi.schedule(date=self.today)
            parsed = []
            for g in raw_games:
                if g.get('status') in ['Final', 'Postponed', 'Cancelled']:
                    continue
                parsed.append({
                    'game_id': g.get('game_id'),
                    'away_team': g.get('away_name'),
                    'home_team': g.get('home_name'),
                    'away_pitcher': g.get('away_probable_pitcher') or 'Unknown Starter',
                    'home_pitcher': g.get('home_probable_pitcher') or 'Unknown Starter'
                })
            return pd.DataFrame(parsed)
        except Exception:
            return pd.DataFrame()

    def scrape_historical_and_savant_data(self):
        """Scrapes advanced baseball tracking layers from historical endpoints."""
        try:
            print("Querying Baseball Savant Hitter & Pitcher Leaderboards...")
            savant_hitters = pb.statcast_batter_expected_stats(self.current_year)
            savant_pitchers = pb.statcast_pitcher_expected_stats(self.current_year)
            
            print("Querying FanGraphs Batting & Pitching Roster Overlays...")
            fg_batting = pb.batting_stats(self.current_year - 1, self.current_year, qual=10)
            fg_pitching = pb.pitching_stats(self.current_year - 1, self.current_year, qual=10)
            
            return {
                'savant_hitters': savant_hitters if savant_hitters is not None else pd.DataFrame(),
                'savant_pitchers': savant_pitchers if savant_pitchers is not None else pd.DataFrame(),
                'fg_batting': fg_batting if fg_batting is not None else pd.DataFrame(),
                'fg_pitching': fg_pitching if fg_pitching is not None else pd.DataFrame()
            }
        except Exception as e:
            print(f"Scraping failed: {e}. Executing with pipeline defaults.")
            return None
    def _compass_to_degrees(self, direction):
        """Translates text wind directions into numerical compass degrees."""
        if isinstance(direction, (int, float)):
            return float(direction)
        mapping = {
            'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
            'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
            'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
            'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
        }
        return mapping.get(str(direction).strip().upper(), 0.0)

    def calculate_environmental_modifier(self, home_team, w_data):
        """Implements the full Delta_Env operational framework formula."""
        p_info = self.park_data.get(home_team, {'factor': 1.00, 'dome': False})
        base_pf = p_info['factor']
        if p_info['dome']:
            return base_pf
            
        elevation_factor = 3.5 if home_team == 'Colorado Rockies' else 1.0
        delta_density = (w_data['temp'] - 70) * 0.001 * elevation_factor
        
        deg_angle = self._compass_to_degrees(w_data['wind_direction'])
        wind_angle_rad = np.radians(deg_angle)
        delta_wind = w_data['wind_speed'] * np.cos(wind_angle_rad) * 0.002
        
        return base_pf * (1 + delta_density + delta_wind)

    def calculate_custom_engine_metrics(self, game, db, delta_env):
        """Calculates precise matchups by contrasting dynamic hitter vs pitcher stats."""
        away_osf, home_osf = 0.320, 0.320 
        away_psi, home_psi = 4.20, 4.20 
        away_bsi, home_bsi = 4.30, 4.30 
        
        prop_baselines = {
            'away_pitcher': {'name': game['away_pitcher'], 'k': 5.2, 'er': 2.4},
            'home_pitcher': {'name': game['home_pitcher'], 'k': 5.2, 'er': 2.4},
            'away_hitters': {'hits': 0.85, 'tb': 1.35, 'rbi': 0.45, 'runs': 0.45, 'hr': 0.12},
            'home_hitters': {'hits': 0.85, 'tb': 1.35, 'rbi': 0.45, 'runs': 0.45, 'hr': 0.12}
        }
        
        away_team, home_team = game['away_team'], game['home_team']
        fg_away_code = self.fg_team_map.get(away_team, 'MIN')
        fg_home_code = self.fg_team_map.get(home_team, 'MIN')

        if db is not None:
            # Batting Base Processing
            fgb = db['fg_batting']
            if not fgb.empty and 'Team' in fgb.columns:
                a_rows = fgb[fgb['Team'] == fg_away_code]
                h_rows = fgb[fgb['Team'] == fg_home_code]
                if not a_rows.empty:
                    away_osf = float(a_rows['wOBA'].mean())
                    prop_baselines['away_hitters'] = {'hits': 0.88, 'tb': 1.42, 'rbi': 0.48, 'runs': 0.48, 'hr': 0.14}
                if not h_rows.empty:
                    home_osf = float(h_rows['wOBA'].mean())
                    prop_baselines['home_hitters'] = {'hits': 0.88, 'tb': 1.42, 'rbi': 0.48, 'runs': 0.48, 'hr': 0.14}

            # Pitcher Base Processing
            fgp = db['fg_pitching']
            if not fgp.empty and 'Name' in fgp.columns and 'FIP' in fgp.columns:
                asp_row = fgp[fgp['Name'].str.contains(game['away_pitcher'].split()[-1], na=False, case=False)] if game['away_pitcher'] != 'Unknown Starter' else pd.DataFrame()
                hsp_row = fgp[fgp['Name'].str.contains(game['home_pitcher'].split()[-1], na=False, case=False)] if game['home_pitcher'] != 'Unknown Starter' else pd.DataFrame()
                
                if not asp_row.empty:
                    away_psi = float(asp_row['FIP'].mean())
                    if 'SO' in asp_row.columns:
                        prop_baselines['away_pitcher']['k'] = (float(asp_row['SO'].mean()) / 32.0) * 5.5
                    if 'ER' in asp_row.columns:
                        prop_baselines['away_pitcher']['er'] = float(asp_row['ER'].mean()) / 32.0 * 5.0
                if not hsp_row.empty:
                    home_psi = float(hsp_row['FIP'].mean())
                    if 'SO' in hsp_row.columns:
                        prop_baselines['home_pitcher']['k'] = (float(hsp_row['SO'].mean()) / 32.0) * 5.5
                    if 'ER' in hsp_row.columns:
                        prop_baselines['home_pitcher']['er'] = float(hsp_row['ER'].mean()) / 32.0 * 5.0

        metrics = {
            'away_osf': away_osf, 'home_osf': home_osf,
            'away_psi': away_psi, 'home_psi': home_psi,
            'away_bsi': away_bsi, 'home_bsi': home_bsi,
            'delta_env': delta_env,
            'away_avg_runs_per_inning': (away_osf * 1.62) * (home_psi / 4.20) * delta_env,
            'home_avg_runs_per_inning': (home_osf * 1.62) * (away_psi / 4.20) * delta_env,
            'props': prop_baselines
        }
        return metrics
    def execute_segment_simulation(self, metrics, num_sims=10000):
        """Vectorizes Monte Carlo paths to generate team scoring and all DraftKings prop matrices."""
        env = np.random.normal(metrics['delta_env'], 0.04, num_sims)
        
        lambda_away_sp = metrics['away_avg_runs_per_inning'] * env
        lambda_home_sp = metrics['home_avg_runs_per_inning'] * env
        
        runs_away = np.zeros((num_sims, 9))
        runs_home = np.zeros((num_sims, 9))
        
        for i in range(9):
            if i < 5:
                runs_away[:, i] = np.random.poisson(lambda_away_sp, num_sims)
                runs_home[:, i] = np.random.poisson(lambda_home_sp, num_sims)
            else:
                runs_away[:, i] = np.random.poisson(lambda_away_sp * 1.05, num_sims)
                runs_home[:, i] = np.random.poisson(lambda_home_sp * 1.05, num_sims)

        # Pitcher Prop Vectors
        k_away_sp = np.random.poisson(metrics['props']['away_pitcher']['k'] / env, num_sims)
        k_home_sp = np.random.poisson(metrics['props']['home_pitcher']['k'] / env, num_sims)
        er_away_sp = np.random.poisson(metrics['props']['away_pitcher']['er'] * env, num_sims)
        er_home_sp = np.random.poisson(metrics['props']['home_pitcher']['er'] * env, num_sims)

        # Batter Prop Vectors
        p_away = metrics['props']['away_hitters']
        p_home = metrics['props']['home_hitters']

        away_hits = np.random.poisson(p_away['hits'] * env, num_sims)
        home_hits = np.random.poisson(p_home['hits'] * env, num_sims)
        away_tb = np.random.poisson(p_away['tb'] * env, num_sims)
        home_tb = np.random.poisson(p_home['tb'] * env, num_sims)
        away_rbi = np.random.poisson(p_away['rbi'] * env, num_sims)
        home_rbi = np.random.poisson(p_home['rbi'] * env, num_sims)
        away_runs = np.random.poisson(p_away['runs'] * env, num_sims)
        home_runs = np.random.poisson(p_home['runs'] * env, num_sims)
        away_hr = np.random.binomial(1, np.clip(p_away['hr'] * env, 0, 1), num_sims)
        home_hr = np.random.binomial(1, np.clip(p_home['hr'] * env, 0, 1), num_sims)

        return {
            'F3': (np.sum(runs_away[:, :3], axis=1), np.sum(runs_home[:, :3], axis=1)),
            'F5': (np.sum(runs_away[:, :5], axis=1), np.sum(runs_home[:, :5], axis=1)),
            'F7': (np.sum(runs_away[:, :7], axis=1), np.sum(runs_home[:, :7], axis=1)),
            'FG': (np.sum(runs_away, axis=1), np.sum(runs_home, axis=1)),
            'pitcher_props': {
                'away_k': k_away_sp, 'home_k': k_home_sp, 'away_er': er_away_sp, 'home_er': er_home_sp
            },
            'batter_props': {
                'away_hits': away_hits, 'home_hits': home_hits, 'away_tb': away_tb, 'home_tb': home_tb,
                'away_rbi': away_rbi, 'home_rbi': home_rbi, 'away_runs': away_runs, 'home_runs': home_runs,
                'away_hr': away_hr, 'home_hr': home_hr
            }
        }
    def _calculate_score_mode(self, away_scores, home_scores):
        """Finds the single most frequent exact score combination from the simulation matrix."""
        df_scores = pd.DataFrame({'away': away_scores.astype(int), 'home': home_scores.astype(int)})
        mode_row = df_scores.value_counts().idxmax()
        return mode_row[0], mode_row[1]

    def compute_market_edges(self, sim_data, game):
        results = []
        segments = {'First 3 Innings': 'F3', 'First 5 Innings': 'F5', 'First 7 Innings': 'F7', 'Full Game': 'FG'}
        for seg_name, key in segments.items():
            away_scores, home_scores = sim_data[key]
            home_ml_prob = np.sum(home_scores > away_scores) / len(home_scores)
            away_ml_prob = np.sum(away_scores > home_scores) / len(away_scores)
            dk_total_line = round(np.mean(away_scores + home_scores) * 2) / 2
            over_prob = np.sum((away_scores + home_scores) > dk_total_line) / len(away_scores)
            
            # Mode processing integration
            mode_away, mode_home = self._calculate_score_mode(away_scores, home_scores)
            
            results.append({
                'Matchup': f"{game['away_team']} @ {game['home_team']}", 'Segment': seg_name,
                'Proj_Score': f"{mode_away} - {mode_home}",
                'Home_ML_Prob': f"{home_ml_prob * 100:.1f}%", 'Away_ML_Prob': f"{away_ml_prob * 100:.1f}%",
                'Target_DK_Total': dk_total_line, 'Over_Probability': f"{over_prob * 100:.1f}%"
            })
        return results

    def process_all_dk_props(self, sim_data, game):
        props_list = []
        p_sim = sim_data['pitcher_props']
        b_sim = sim_data['batter_props']
        
        # 1. Map Pitcher Props Markets (Strikeouts & Earned Runs)
        pitchers = [('away', game['away_pitcher'], game['away_team']), ('home', game['home_pitcher'], game['home_team'])]
        for side, name, team in pitchers:
            for prop_key, label in [('k', 'Strikeouts (O/U)'), ('er', 'Earned Runs (O/U)')]:
                arr = p_sim[f"{side}_{prop_key}"]
                mean_val = np.mean(arr)
                line = round(mean_val * 2) / 2
                props_list.append({
                    'Player/Lineup': name, 'Team': team, 'Market_Type': label,
                    'DraftKings_Line': line, 'Simulated_Mean': f"{mean_val:.2f}",
                    'Over_Probability': f"{np.sum(arr > line) / len(arr) * 100:.1f}%"
                })

        # 2. Map Lineup Batter Props Markets (Hits, Total Bases, RBIs, Runs, Home Runs)
        batters = [('away', 'Away Lineup Average', game['away_team']), ('home', 'Home Lineup Average', game['home_team'])]
        b_markets = [('hits', 'Hits (O/U)', 0.5), ('tb', 'Total Bases (O/U)', 1.5), ('rbi', 'RBIs (O/U)', 0.5), ('runs', 'Runs Scored (O/U)', 0.5), ('hr', 'To Hit a Home Run (Yes)', 0.5)]
        for side, name, team in batters:
            for key, label, dk_std_line in b_markets:
                arr = b_sim[f"{side}_{key}"]
                mean_val = np.mean(arr)
                props_list.append({
                    'Player/Lineup': name, 'Team': team, 'Market_Type': label,
                    'DraftKings_Line': dk_std_line, 'Simulated_Mean': f"{mean_val:.2f}",
                    'Over_Probability': f"{np.sum(arr > dk_std_line) / len(arr) * 100:.1f}%"
                })
        return props_list

    async def run_pipeline(self):
        schedule_df = self.fetch_live_schedule()
        if schedule_df.empty:
            print("No active games scheduled for today.")
            return
            
        metrics_db = self.scrape_historical_and_savant_data()
        all_markets, all_props = [], []

        for _, game in schedule_df.iterrows():
            home_team = game['home_team']
            city = self.team_cities.get(home_team, 'New York')
            
            weather_snapshot = await self.get_live_weather(city)
            print(f"Matchup: {game['away_team']} @ {home_team} | Weather: {weather_snapshot}")
            
            delta_env = self.calculate_environmental_modifier(home_team, weather_snapshot)
            computed_matrix = self.calculate_custom_engine_metrics(game, metrics_db, delta_env)
            sim_data = self.execute_segment_simulation(computed_matrix)
            
            all_markets.extend(self.compute_market_edges(sim_data, game))
            all_props.extend(self.process_all_dk_props(sim_data, game))

        os.makedirs('data/predictions', exist_ok=True)
        pd.DataFrame(all_markets).to_csv(f"data/predictions/mlb_market_segments_{self.today}.csv", index=False)
        pd.DataFrame(all_props).to_csv(f"data/predictions/mlb_dk_props_{self.today}.csv", index=False)
        print("Data compilation successfully saved to repository.")

if __name__ == "__main__":
    engine = MLBFullPropQuantEngine()
    asyncio.run(engine.run_pipeline())
