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

class MLBInningByInningEngine:
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
        """Ingests current day games along with individual roster lineups directly from MLB StatsAPI."""
        try:
            raw_games = statsapi.schedule(date=self.today)
            parsed = []
            for g in raw_games:
                if g.get('status') in ['Final', 'Postponed', 'Cancelled']:
                    continue
                gid = g.get('game_id')
                
                away_batters, home_batters = [], []
                try:
                    box = statsapi.boxscore_data(gid)
                    away_batters = [b['person']['fullName'] for b in box['awayBatters'] if 'person' in b]
                    home_batters = [b['person']['fullName'] for b in box['homeBatters'] if 'person' in b]
                except Exception:
                    pass
                
                parsed.append({
                    'game_id': gid,
                    'away_team': g.get('away_name'),
                    'home_team': g.get('home_name'),
                    'away_pitcher': g.get('away_probable_pitcher') or 'Unknown Starter',
                    'home_pitcher': g.get('home_probable_pitcher') or 'Unknown Starter',
                    'away_lineup': away_batters,
                    'home_lineup': home_batters
                })
            return pd.DataFrame(parsed)
        except Exception:
            return pd.DataFrame()

    def scrape_historical_and_savant_data(self):
        """Scrapes advanced baseball tracking layers directly from Savant to bypass FanGraphs 403 blocks."""
        try:
            print("Querying Official Baseball Savant Statcast Metrics...")
            savant_hitters = pb.statcast_batter_expected_stats(self.current_year)
            savant_pitchers = pb.statcast_pitcher_expected_stats(self.current_year)
            
            # --- STANDARDIZE BATTER COLUMN LAYOUTS ---
            if savant_hitters is not None and not savant_hitters.empty:
                for col in ['last_name, first_name', 'name']:
                    if col in savant_hitters.columns:
                        savant_hitters = savant_hitters.rename(columns={col: 'clean_p_name'})
                if 'clean_p_name' in savant_hitters.columns:
                    savant_hitters['clean_p_name'] = savant_hitters['clean_p_name'].str.replace(r'[^a-zA-Z\s,]', '', regex=True)

            # --- STANDARDIZE PITCHER COLUMN LAYOUTS ---
            if savant_pitchers is not None and not savant_pitchers.empty:
                for col in ['last_name, first_name', 'name', 'player_name']:
                    if col in savant_pitchers.columns:
                        savant_pitchers = savant_pitchers.rename(columns={col: 'clean_p_name'})
                if 'clean_p_name' in savant_pitchers.columns:
                    savant_pitchers['clean_p_name'] = savant_pitchers['clean_p_name'].str.replace(r'[^a-zA-Z\s,]', '', regex=True)
            
            return {
                'savant_hitters': savant_hitters if savant_hitters is not None else pd.DataFrame(),
                'savant_pitchers': savant_pitchers if savant_pitchers is not None else pd.DataFrame(),
                'fg_batting': pd.DataFrame(),
                'fg_pitching': pd.DataFrame()
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

    def _calculate_log5_intersect(self, batter_stat, pitcher_stat, league_avg):
        """Applies Bill James' Log5 formula to isolate true Batter vs Pitcher odds probabilities."""
        numerator = (batter_stat * pitcher_stat) / league_avg
        denominator = numerator + ((1.0 - batter_stat) * (1.0 - pitcher_stat)) / (1.0 - league_avg)
        if denominator == 0:
            return batter_stat
        return numerator / denominator

    def calculate_custom_engine_metrics(self, game, db, delta_env):
        """Calculates precise matchups by running Log5 odds intersections on normalized Savant metrics."""
        league_woba_avg = 0.315
        league_fip_avg = 4.20
        
        away_team, home_team = game['away_team'], game['home_team']
        away_sp, home_sp = game['away_pitcher'], game['home_pitcher']
        
        a_lineup = game['away_lineup'] if game['away_lineup'] else [f"Away Batter {i}" for i in range(1, 10)]
        h_lineup = game['home_lineup'] if game['home_lineup'] else [f"Home Batter {i}" for i in range(1, 10)]
        
        prop_baselines = {
            'away_pitcher': {'name': away_sp, 'k': 5.5, 'er': 2.2},
            'home_pitcher': {'name': home_sp, 'k': 5.5, 'er': 2.2},
            'batters': []
        }
        
        away_team_woba, home_team_woba = league_woba_avg, league_woba_avg
        away_sp_fip, home_sp_fip = league_fip_avg, league_fip_avg
        away_bp_fip, home_bp_fip = league_fip_avg + 0.15, league_fip_avg + 0.15

        if db is not None:
            sb_h = db['savant_hitters']
            if not sb_h.empty and 'team_name' in sb_h.columns and 'est_woba' in sb_h.columns:
                s_away = sb_h[sb_h['team_name'].str.contains(away_team.split()[-1], na=False, case=False)]
                s_home = sb_h[sb_h['team_name'].str.contains(home_team.split()[-1], na=False, case=False)]
                if not s_away.empty: away_team_woba = float(s_away['est_woba'].mean())
                if not s_home.empty: home_team_woba = float(s_home['est_woba'].mean())

            sb_p = db['savant_pitchers']
            if not sb_p.empty and 'clean_p_name' in sb_p.columns and 'est_woba' in sb_p.columns:
                asp_last = away_sp.split()[-1] if len(away_sp.split()) > 0 else 'UNKNOWN_TOKEN'
                hsp_last = home_sp.split()[-1] if len(home_sp.split()) > 0 else 'UNKNOWN_TOKEN'
                
                p_asp = sb_p[sb_p['clean_p_name'].str.contains(asp_last, na=False, case=False)] if away_sp != 'Unknown Starter' else pd.DataFrame()
                p_hsp = sb_p[sb_p['clean_p_name'].str.contains(hsp_last, na=False, case=False)] if home_sp != 'Unknown Starter' else pd.DataFrame()
                
                if not p_asp.empty:
                    away_sp_fip = float(p_asp['est_woba'].mean()) * 12.5
                    prop_baselines['away_pitcher']['k'] = 6.0 if float(p_asp['est_woba'].mean()) < 0.300 else 5.0
                if not p_hsp.empty:
                    home_sp_fip = float(p_hsp['est_woba'].mean()) * 12.5
                    prop_baselines['home_pitcher']['k'] = 6.0 if float(p_hsp['est_woba'].mean()) < 0.300 else 5.0

        # Run Log5 Team scoring matrices
        away_runs_vs_sp = (self._calculate_log5_intersect(away_team_woba, home_sp_fip/12.5, league_woba_avg) * 1.62) * delta_env
        home_runs_vs_sp = (self._calculate_log5_intersect(home_team_woba, away_sp_fip/12.5, league_woba_avg) * 1.62) * delta_env
        away_runs_vs_bp = (self._calculate_log5_intersect(away_team_woba, home_bp_fip/12.5, league_woba_avg) * 1.62) * delta_env
        home_runs_vs_bp = (self._calculate_log5_intersect(home_team_woba, away_bp_fip/12.5, league_woba_avg) * 1.62) * delta_env

        # 3. Match Individual Hitters by Last Name tokens
        if db is not None and not sb_h.empty and 'clean_p_name' in sb_h.columns:
            for side, lineup, team_label, opp_sp_fip in [('away', a_lineup, away_team, home_sp_fip), ('home', h_lineup, home_team, away_sp_fip)]:
                for player_name in lineup[:9]:
                    pHits, pTB, pRBI, pRuns, pHR = 0.85, 1.35, 0.45, 0.45, 0.12
                    p_last = player_name.split()[-1] if len(player_name.split()) > 0 else 'UNKNOWN_TOKEN'
                    
                    p_row = sb_h[sb_h['clean_p_name'].str.contains(p_last, na=False, case=False)]
                    if not p_row.empty and 'est_woba' in p_row.columns:
                        hitter_xwoba = float(p_row['est_woba'].mean())
                        matched_woba = self._calculate_log5_intersect(hitter_xwoba, opp_sp_fip/12.5, league_woba_avg)
                        pHits = 0.85 * (matched_woba / league_woba_avg)
                        pTB = 1.35 * (matched_woba / league_woba_avg)
                        pHR = 0.12 * (matched_woba / league_woba_avg)

                    prop_baselines['batters'].append({
                        'name': player_name, 'side': side, 'team': team_label,
                        'hits': pHits, 'tb': pTB, 'rbi': pRBI, 'runs': pRuns, 'hr': pHR
                    })

        metrics = {
            'away_osf': away_team_woba, 'home_osf': home_team_woba,
            'away_psi': away_sp_fip, 'home_psi': home_sp_fip,
            'away_bsi': away_bp_fip, 'home_bsi': home_bp_fip,
            'delta_env': delta_env,
            'lambda_away_sp': away_runs_vs_sp, 'lambda_home_sp': home_runs_vs_sp,
            'lambda_away_rp': away_runs_vs_bp, 'lambda_home_rp': home_runs_vs_bp,
            'props': prop_baselines
        }
        return metrics
    def execute_segment_simulation(self, metrics, num_sims=10000):
        """Vectorizes Monte Carlo paths to generate inning-by-inning team scoring matrices and player props."""
        env = np.random.normal(metrics['delta_env'], 0.04, num_sims)
        
        l_away_sp = metrics['lambda_away_sp'] * env
        l_home_sp = metrics['lambda_home_sp'] * env
        l_away_rp = metrics['lambda_away_rp'] * env
        l_home_rp = metrics['lambda_home_rp'] * env

        runs_away = np.zeros((num_sims, 9))
        runs_home = np.zeros((num_sims, 9))
        
        for i in range(9):
            if i < 5:
                runs_away[:, i] = np.random.poisson(l_away_sp, num_sims)
                runs_home[:, i] = np.random.poisson(l_home_sp, num_sims)
            else:
                runs_away[:, i] = np.random.poisson(l_away_rp, num_sims)
                runs_home[:, i] = np.random.poisson(l_home_rp, num_sims)

        k_away = np.random.poisson(metrics['props']['away_pitcher']['k'] / env, num_sims)
        k_home = np.random.poisson(metrics['props']['home_pitcher']['k'] / env, num_sims)
        er_away = np.random.poisson(metrics['props']['away_pitcher']['er'] * env, num_sims)
        er_home = np.random.poisson(metrics['props']['home_pitcher']['er'] * env, num_sims)

        simulated_batters = []
        for b in metrics['props']['batters']:
            simulated_batters.append({
                'name': b['name'], 'team': b['team'],
                'hits': np.random.poisson(b['hits'] * env, num_sims),
                'tb': np.random.poisson(b['tb'] * env, num_sims),
                'rbi': np.random.poisson(b['rbi'] * env, num_sims),
                'runs': np.random.poisson(b['runs'] * env, num_sims),
                'hr': np.random.binomial(1, np.clip(b['hr'] * env, 0, 1), num_sims)
            })

        return {
            'F3': (np.sum(runs_away[:, :3], axis=1), np.sum(runs_home[:, :3], axis=1)),
            'F5': (np.sum(runs_away[:, :5], axis=1), np.sum(runs_home[:, :5], axis=1)),
            'F7': (np.sum(runs_away[:, :7], axis=1), np.sum(runs_home[:, :7], axis=1)),
            'FG': (np.sum(runs_away, axis=1), np.sum(runs_home, axis=1)),
            'pitcher_props': {'away_k': k_away, 'home_k': k_home, 'away_er': er_away, 'home_er': er_home},
            'batter_props': simulated_batters
        }
    def _calculate_score_mode(self, away_scores, home_scores):
        """Finds the single most frequent exact score combination and unpacks clean integers."""
        df = pd.DataFrame({'away': away_scores.astype(int), 'home': home_scores.astype(int)})
        mode_tuple = df.value_counts().idxmax()
        return int(mode_tuple[0]), int(mode_tuple[1])

    def _calculate_array_mode(self, data_array):
        return int(pd.Series(data_array.astype(int)).value_counts().idxmax())

    def compute_market_edges(self, sim_data, game):
        results = []
        segments = {'First 3 Innings': 'F3', 'First 5 Innings': 'F5', 'First 7 Innings': 'F7', 'Full Game': 'FG'}
        for seg_name, key in segments.items():
            away_scores, home_scores = sim_data[key]
            
            resolved = np.sum(home_scores != away_scores)
            home_ml_prob = np.sum(home_scores > away_scores) / resolved if resolved > 0 else 0.50
            away_ml_prob = np.sum(away_scores > home_scores) / resolved if resolved > 0 else 0.50
            
            dk_total_line = round(np.mean(away_scores + home_scores) * 2) / 2
            over_prob = np.sum((away_scores + home_scores) > dk_total_line) / len(away_scores)
            
            mode_a, mode_h = self._calculate_score_mode(away_scores, home_scores)
            
            results.append({
                'Matchup': f"{game['away_team']} @ {game['home_team']}", 'Segment': seg_name,
                'Proj_Score': f"{mode_a} - {mode_h}",
                'Home_ML_Probability': f"{home_ml_prob * 100:.1f}%", 'Away_ML_Probability': f"{away_ml_prob * 100:.1f}%",
                'Target_DK_Total_Line': dk_total_line, 'Over_Total_Probability': f"{over_prob * 100:.1f}%", 'Under_Total_Probability': f"{(1.0 - over_prob) * 100:.1f}%"
            })
        return results

    def process_all_dk_props(self, sim_data, game):
        props_list = []
        p_sim = sim_data['pitcher_props']
        
        pitchers = [('away', game['away_pitcher'], game['away_team']), ('home', game['home_pitcher'], game['home_team'])]
        for side, name, team in pitchers:
            for p_key, label in [('k', 'Strikeouts (O/U)'), ('er', 'Earned Runs (O/U)')]:
                arr = p_sim[f"{side}_{p_key}"]
                m_outcome = self._calculate_array_mode(arr)
                over_p = np.sum(arr > m_outcome) / len(arr)
                props_list.append({
                    'Player_Name': name, 'Team': team, 'Market_Type': label, 'Most_Likely_Line_Outcome': m_outcome,
                    'Over_Probability': f"{over_p * 100:.1f}%", 'Under_Probability': f"{(1.0 - over_p) * 100:.1f}%"
                })

        b_markets = [('hits', 'Hits (O/U)'), ('tb', 'Total Bases (O/U)'), ('rbi', 'RBIs (O/U)'), ('runs', 'Runs Scored (O/U)'), ('hr', 'Home Runs (O/U)')]
        for b_data in sim_data['batter_props']:
            for key, label in b_markets:
                arr = b_data[key]
                m_outcome = self._calculate_array_mode(arr)
                over_p = np.sum(arr > m_outcome) / len(arr)
                props_list.append({
                    'Player_Name': b_data['name'], 'Team': b_data['team'], 'Market_Type': label, 'Most_Likely_Line_Outcome': m_outcome,
                    'Over_Probability': f"{over_p * 100:.1f}%", 'Under_Probability': f"{(1.0 - over_p) * 100:.1f}%"
                })
        return props_list

    async def run_pipeline(self):
        schedule_df = self.fetch_live_schedule()
        if schedule_df.empty:
            print("No active matching games available on today's schedule.")
            return
            
        metrics_db = self.scrape_historical_and_savant_data()
        all_segments_out, all_props_out = [], []

        for _, game in schedule_df.iterrows():
            away_sp = game.get('away_pitcher', 'Unknown Starter')
            home_sp = game.get('home_pitcher', 'Unknown Starter')
            away_team = game['away_team']
            home_team = game['home_team']

            if away_sp == 'Unknown Starter' or home_sp == 'Unknown Starter':
                print(f"⚠️ [SKIPPED] {away_team} @ {home_team} - Simulation aborted due to unconfirmed Starting Pitcher.")
                continue

            city = self.team_cities.get(home_team, 'New York')
            weather_snapshot = await self.get_live_weather(city)
            delta_env = self.calculate_environmental_modifier(home_team, weather_snapshot)
            computed_matrix = self.calculate_custom_engine_metrics(game, metrics_db, delta_env)
            
            print(f"🚀 Executing 10,000-Run Inning-by-Inning Simulation for {away_team} @ {home_team}...")
            sim_data = self.execute_segment_simulation(computed_matrix)
            
            all_segments_out.extend(self.compute_market_edges(sim_data, game))
            all_props_out.extend(self.process_all_dk_props(sim_data, game))

        os.makedirs('data/predictions', exist_ok=True)
        pd.DataFrame(all_segments_out).to_csv(f"data/predictions/mlb_market_segments_{self.today}.csv", index=False)
        pd.DataFrame(all_props_out).to_csv(f"data/predictions/mlb_dk_props_{self.today}.csv", index=False)
        print("Data compilation successfully saved to repository.")

if __name__ == "__main__":
    engine = MLBInningByInningEngine()
    asyncio.run(engine.run_pipeline())
