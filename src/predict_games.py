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

class MLBPitchByPitchEngine:
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
        """Ingests current day games along with individual roster lineups directly from StatsAPI."""
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
        """Scrapes advanced baseball tracking layers from historical endpoints."""
        try:
            print("Querying Baseball Savant Leaderboards...")
            savant_hitters = pb.statcast_batter_expected_stats(self.current_year)
            savant_pitchers = pb.statcast_pitcher_expected_stats(self.current_year)
            
            print("Querying FanGraphs Batting & Pitching Roster Overlays...")
            fg_batting = pb.batting_stats(self.current_year - 1, self.current_year, qual=5)
            fg_pitching = pb.pitching_stats(self.current_year - 1, self.current_year, qual=5)
            
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
        """Builds granular pitch-by-pitch probabilities for individual player matchups."""
        away_team, home_team = game['away_team'], game['home_team']
        away_sp, home_sp = game['away_pitcher'], game['home_pitcher']
        
        a_lineup = game['away_lineup'] if game['away_lineup'] else [f"Away Batter {i}" for i in range(1, 10)]
        h_lineup = game['home_lineup'] if game['home_lineup'] else [f"Home Batter {i}" for i in range(1, 10)]
        
        # Enforce pitch-level probability default structures
        pitchers = {
            'away': {'name': away_sp, 'strike_pct': 0.64, 'walk_pct': 0.08, 'in_play_fip': 4.10, 'fatigue_max': 95},
            'home': {'name': home_sp, 'strike_pct': 0.65, 'walk_pct': 0.07, 'in_play_fip': 3.95, 'fatigue_max': 95},
            'away_bullpen': {'strike_pct': 0.63, 'walk_pct': 0.09, 'in_play_fip': 4.20},
            'home_bullpen': {'strike_pct': 0.64, 'walk_pct': 0.08, 'in_play_fip': 4.10}
        }
        
        batters = []
        fgb = db['fg_batting'] if db is not None else pd.DataFrame()
        fgp = db['fg_pitching'] if db is not None else pd.DataFrame()

        # Extract Pitcher Statistics
        if not fgp.empty and 'Name' in fgp.columns:
            for side, name in [('away', away_sp), ('home', home_sp)]:
                p_row = fgp[fgp['Name'].str.contains(name.split()[-1], na=False, case=False)] if len(name.split()) > 0 else pd.DataFrame()
                if not p_row.empty:
                    pitchers[side]['strike_pct'] = float(p_row['Zone%'].mean()) / 100.0 if 'Zone%' in p_row.columns else 0.64
                    pitchers[side]['walk_pct'] = float(p_row['BB%'].mean()) / 100.0 if 'BB%' in p_row.columns else 0.08
                    pitchers[side]['in_play_fip'] = float(p_row['FIP'].mean())

        # Extract Individual Batter Statistics
        for side, lineup, team in [('away', a_lineup, away_team), ('home', h_lineup, home_team)]:
            for i, p_name in enumerate(lineup[:9]):
                b_stats = {'name': p_name, 'side': side, 'team': team, 'slot': i+1, 'contact_pct': 0.78, 'single_ratio': 0.65, 'double_ratio': 0.18, 'triple_ratio': 0.02, 'hr_ratio': 0.15}
                
                if not fgb.empty and 'Name' in fgb.columns:
                    b_row = fgb[fgb['Name'].str.contains(p_name.split()[-1], na=False, case=False)] if len(p_name.split()) > 0 else pd.DataFrame()
                    if not b_row.empty:
                        b_stats['contact_pct'] = float(b_row['Contact%'].mean()) / 100.0 if 'Contact%' in b_row.columns else 0.78
                        total_hits = float(b_row['H'].sum()) if 'H' in b_row.columns and b_row['H'].sum() > 0 else 1.0
                        b_stats['single_ratio'] = (total_hits - float(b_row['2B'].sum() + b_row['3B'].sum() + b_row['HR'].sum())) / total_hits
                        b_stats['double_ratio'] = float(b_row['2B'].sum()) / total_hits
                        b_stats['triple_ratio'] = float(b_row['3B'].sum()) / total_hits
                        b_stats['hr_ratio'] = float(b_row['HR'].sum()) / total_hits

                batters.append(b_stats)

        return {'pitchers': pitchers, 'batters': pd.DataFrame(batters), 'delta_env': delta_env}
    def _simulate_single_game_pbp(self, metrics):
        """Executes a true pitch-by-pitch stochastic simulation path for 9 full innings."""
        pitchers = metrics['pitchers']
        batters_df = metrics['batters']
        env = metrics['delta_env']
        
        runs = {'away': 0, 'home': 0}
        inning_scores = {'F3': {'away': 0, 'home': 0}, 'F5': {'away': 0, 'home': 0}, 'F7': {'away': 0, 'home': 0}, 'FG': {'away': 0, 'home': 0}}
        props = {'away_pitcher_k': 0, 'home_pitcher_k': 0, 'away_pitcher_er': 0, 'home_pitcher_er': 0, 'player_events': {}}
        
        pitch_counts = {'away_sp': 0, 'home_sp': 0}
        lineup_index = {'away': 0, 'home': 0}

        for inning in range(1, 10):
            for half in ['top', 'bottom']:
                hitting_side = 'away' if half == 'top' else 'home'
                pitching_side = 'home' if half == 'top' else 'away'
                
                # Active Base Runner Tracking Array: [First, Second, Third]
                bases = [0, 0, 0]
                outs = 0
                
                while outs < 3:
                    # Roster Lineup Management Loops
                    side_batters = batters_df[batters_df['side'] == hitting_side].reset_index(drop=True)
                    idx = lineup_index[hitting_side]
                    batter = side_batters.iloc[idx % 9]
                    b_name = batter['name']
                    
                    if b_name not in props['player_events']:
                        props['player_events'][b_name] = {'hits': 0, 'tb': 0, 'rbi': 0, 'runs': 0, 'hr': 0}
                    
                    # Bullpen Dependency Routing Matrix
                    is_sp = True
                    sp_label = f"{pitching_side}_sp"
                    if pitch_counts.get(sp_label, 0) >= pitchers[pitching_side]['fatigue_max']:
                        is_sp = False
                        p_profile = pitchers[f"{pitching_side}_bullpen"]
                    else:
                        p_profile = pitchers[pitching_side]
                        pitch_counts[sp_label] += 1
                    
                    # 1. Reset Count State for New Plate Appearance
                    balls, strikes = 0, 0
                    pa_resolved = False
                    
                    while not pa_resolved:
                        if is_sp: pitch_counts[sp_label] += 1
                        
                        # Pitch Multi-Class Decision Matrix
                        rand_pitch = np.random.rand()
                        strike_threshold = p_profile['strike_pct']
                        
                        if rand_pitch < strike_threshold:
                            # Swing Decision Logic
                            if np.random.rand() < batter['contact_pct']:
                                # Ball put in play
                                pa_resolved = True
                                rand_hit = np.random.rand() * env
                                
                                # Compare against contact profiles
                                if rand_hit < (1.0 / p_profile['in_play_fip'] * 2.2):
                                    props['player_events'][b_name]['hits'] += 1
                                    hit_type = np.random.rand()
                                    
                                    # Advance Base Runners dynamically
                                    if hit_type < batter['single_ratio']:
                                        props['player_events'][b_name]['tb'] += 1
                                        new_runs = bases[2]
                                        bases = [1, bases[0], bases[1]]
                                    elif hit_type < (batter['single_ratio'] + batter['double_ratio']):
                                        props['player_events'][b_name]['tb'] += 2
                                        new_runs = bases[2] + bases[1]
                                        bases = [0, 1, bases[0]]
                                    elif hit_type < (1.0 - batter['hr_ratio']):
                                        props['player_events'][b_name]['tb'] += 3
                                        new_runs = bases[2] + bases[1] + bases[0]
                                        bases = [0, 0, 1]
                                    else:
                                        props['player_events'][b_name]['tb'] += 4
                                        props['player_events'][b_name]['hr'] += 1
                                        new_runs = 1 + bases[2] + bases[1] + bases[0]
                                        bases = [0, 0, 0]
                                        
                                    runs[hitting_side] += new_runs
                                    props['player_events'][b_name]['rbi'] += new_runs
                                    if is_sp: props[f"{pitching_side}_pitcher_er"] += new_runs
                                else:
                                    outs += 1
                            else:
                                strikes += 1
                                if strikes == 3:
                                    pa_resolved = True
                                    outs += 1
                                    if is_sp: props[f"{pitching_side}_pitcher_k"] += 1
                        else:
                            if np.random.rand() < 0.05: # Foul ball check
                                if strikes < 2: strikes += 1
                            else:
                                balls += 1
                                if balls == 4:
                                    pa_resolved = True
                                    # Walk runner advancement matrix
                                    if bases[0] == 1:
                                        if bases[1] == 1:
                                            if bases[2] == 1:
                                                runs[hitting_side] += 1
                                                if is_sp: props[f"{pitching_side}_pitcher_er"] += 1
                                            bases[2] = 1
                                        bases[1] = 1
                                    bases[0] = 1

                    lineup_index[hitting_side] += 1
                    
            # Log segment checkpoints at the end of targeted frames
            if inning == 3:
                inning_scores['F3'] = runs.copy()
            elif inning == 5:
                inning_scores['F5'] = runs.copy()
            elif inning == 7:
                inning_scores['F7'] = runs.copy()
                
        inning_scores['FG'] = runs.copy()
        return inning_scores, props
    def _calculate_score_mode(self, away_scores, home_scores):
        df = pd.DataFrame({'away': away_scores, 'home': home_scores})
        return df.value_counts().idxmax(), df.value_counts().idxmax()

    def _calculate_array_mode(self, data_list):
        return int(pd.Series(data_list).value_counts().idxmax())

    def run_monte_carlo(self, metrics, num_sims=1000):
        all_markets, pitcher_k_away, pitcher_k_home, pitcher_er_away, pitcher_er_home = [], [], [], [], []
        segment_data = {'F3': ([], []), 'F5': ([], []), 'F7': ([], []), 'FG': ([], [])}
        hitter_aggregates = {}

        for _ in range(num_sims):
            scores, props = self._simulate_single_game_pbp(metrics)
            
            for k in ['F3', 'F5', 'F7', 'FG']:
                segment_data[k][0].append(scores[k]['away'])
                segment_data[k][1].append(scores[k]['home'])
                
            pitcher_k_away.append(props['away_pitcher_k'])
            pitcher_k_home.append(props['home_pitcher_k'])
            pitcher_er_away.append(props['away_pitcher_er'])
            pitcher_er_home.append(props['home_pitcher_er'])
            
            for player, events in props['player_events'].items():
                if player not in hitter_aggregates:
                    hitter_aggregates[player] = {'hits': [], 'tb': [], 'rbi': [], 'runs': [], 'hr': []}
                for m in ['hits', 'tb', 'rbi', 'runs', 'hr']:
                    hitter_aggregates[player][m].append(events[m])

        return segment_data, pitcher_k_away, pitcher_k_home, pitcher_er_away, pitcher_er_home, hitter_aggregates

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

            # 🛑 CRITICAL SAFETY CHECK: Skip game if either starting pitcher is unannounced
            if away_sp == 'Unknown Starter' or home_sp == 'Unknown Starter':
                print(f"⚠️ [SKIPPED] {away_team} @ {home_team} - Simulation aborted due to unconfirmed Starting Pitcher line.")
                continue

            city = self.team_cities.get(home_team, 'New York')
            weather_snapshot = await self.get_live_weather(city)
            delta_env = self.calculate_environmental_modifier(home_team, weather_snapshot)
            computed_matrix = self.calculate_custom_engine_metrics(game, metrics_db, delta_env)
            
            print(f"🚀 Executing 1,000-Run Pitch-by-Pitch Simulation for {away_team} @ {home_team} ({away_sp} vs {home_sp})...")
            seg_data, k_a, k_h, er_a, er_h, h_aggr = self.run_monte_carlo(computed_matrix)
            
            # 1. Translate Segment Moneylines & Totals
            for label, key in [('First 3 Innings', 'F3'), ('First 5 Innings', 'F5'), ('First 7 Innings', 'F7'), ('Full Game', 'FG')]:
                a_scores = np.array(seg_data[key][0])
                h_scores = np.array(seg_data[key][1])
                mode_pair = self._calculate_score_mode(a_scores, h_scores)
                over_prob = np.sum((a_scores + h_scores) > (round(np.mean(a_scores + h_scores) * 2) / 2)) / len(a_scores)
                
                all_segments_out.append({
                    'Matchup': f"{away_team} @ {home_team}", 'Segment': label, 'Proj_Score': f"{mode_pair[0][0]} - {mode_pair[0][1]}",
                    'Home_ML_Probability': f"{(np.sum(h_scores > a_scores)/len(a_scores))*100:.1f}%",
                    'Away_ML_Probability': f"{(np.sum(a_scores > h_scores)/len(a_scores))*100:.1f}%",
                    'Over_Probability': f"{over_prob * 100:.1f}%", 'Under_Probability': f"{(1.0 - over_prob) * 100:.1f}%"
                })

            # 2. Translate Named Pitcher Profiles
            for name, team, k_arr, er_arr in [(away_sp, away_team, k_a, er_a), (home_sp, home_team, k_h, er_h)]:
                for arr, label in [(k_arr, 'Strikeouts (O/U)'), (er_arr, 'Earned Runs (O/U)')]:
                    mode_line = self._calculate_array_mode(np.array(arr))
                    over_p = np.sum(np.array(arr) > mode_line) / len(arr)
                    all_props_out.append({
                        'Player_Name': name, 'Team': team, 'Market_Type': label, 'Most_Likely_Line_Outcome': mode_line,
                        'Over_Probability': f"{over_p*100:.1f}%", 'Under_Probability': f"{(1.0 - over_p)*100:.1f}%"
                    })

            # 3. Translate Named Hitter Profiles
            for p_name, datasets in h_aggr.items():
                for market_key, label in [('hits', 'Hits (O/U)'), ('tb', 'Total Bases (O/U)'), ('rbi', 'RBIs (O/U)'), ('runs', 'Runs Scored (O/U)'), ('hr', 'Home Runs (O/U)')]:
                    arr = np.array(datasets[market_key])
                    mode_line = self._calculate_array_mode(arr)
                    over_p = np.sum(arr > mode_line) / len(arr)
                    all_props_out.append({
                        'Player_Name': p_name, 'Team': home_team if p_name in game['home_lineup'] else away_team,
                        'Market_Type': label, 'Most_Likely_Line_Outcome': mode_line,
                        'Over_Probability': f"{over_p*100:.1f}%", 'Under_Probability': f"{(1.0 - over_p)*100:.1f}%"
                    })

        os.makedirs('data/predictions', exist_ok=True)
        pd.DataFrame(all_segments_out).to_csv(f"data/predictions/mlb_market_segments_{self.today}.csv", index=False)
        pd.DataFrame(all_props_out).to_csv(f"data/predictions/mlb_dk_props_{self.today}.csv", index=False)
        print("Data compilation successfully saved to repository.")

if __name__ == "__main__":
    engine = MLBFullIndividualPropEngine()
    asyncio.run(engine.run_pipeline())
