import os
import glob
import datetime
import numpy as np
import pandas as pd
import statsapi

class MLBBacktestEngine:
    def __init__(self, unit_size=100.0):
        # Establish flat bet sizing thresholds ($100 per bet)
        self.unit_size = unit_size
        self.predictions_path = "data/predictions/mlb_market_segments_*.csv"
        
        # Standard bookmaker juice configurations (-110 odds parameters)
        self.standard_juiced_payout = 100.0 / 110.0
        self.moneyline_payout_multiplier = 1.00
    def _fetch_actual_results(self, target_date):
        """Queries MLB StatsAPI to grab official game scores for verification."""
        print(f"Retrieving official structural box scores for {target_date}...")
        results = {}
        try:
            games = statsapi.schedule(date=target_date)
            for g in games:
                if g.get('status') != 'Final':
                    continue
                away = g['away_name']
                home = g['home_name']
                matchup_key = f"{away} @ {home}"
                
                results[matchup_key] = {
                    'away_score': int(g['away_score']),
                    'home_score': int(g['home_score']),
                    'total_runs': int(g['away_score'] + g['home_score'])
                }
            return results
        except Exception as e:
            print(f"Error fetching actual score results: {e}")
            return {}
    def _evaluate_bets(self, pred_row, actual):
        """Applies true sports betting settlement matrix constraints with flexible column string parsing."""
        def get_row_value(row, choices, default="0%"):
            for choice in choices:
                if choice in row.index:
                    return str(row[choice])
            return default

        raw_home_ml = get_row_value(pred_row, ['Home_ML_Probability', 'Home_ML_Prob'])
        raw_away_ml = get_row_value(pred_row, ['Away_ML_Probability', 'Away_ML_Prob'])
        raw_over = get_row_value(pred_row, ['Over_Total_Probability', 'Over_Probability'])
        
        p_home_ml = float(raw_home_ml.replace('%', '')) / 100.0
        p_away_ml = float(raw_away_ml.replace('%', '')) / 100.0
        p_over = float(raw_over.replace('%', '')) / 100.0
        
        line_total = 8.5
        for total_key in ['Target_DK_Total_Line', 'Target_DK_Total', 'DK_Total_Line']:
            if total_key in pred_row.index:
                line_total = float(pred_row[total_key])
                break
        
        act_away = actual['away_score']
        act_home = actual['home_score']
        act_total = actual['total_runs']
        
        wagers = []
        
        # Moneyline execution (+EV threshold set to 54%)
        if p_home_ml > 0.54:
            win = 1.0 if act_home > act_away else 0.0
            profit = self.unit_size * self.moneyline_payout_multiplier if win else -self.unit_size
            wagers.append({'Market': 'Moneyline', 'Selection': 'Home', 'Win': win, 'Profit': profit})
        elif p_away_ml > 0.54:
            win = 1.0 if act_away > act_home else 0.0
            profit = self.unit_size * self.moneyline_payout_multiplier if win else -self.unit_size
            wagers.append({'Market': 'Moneyline', 'Selection': 'Away', 'Win': win, 'Profit': profit})
            
        # Totals Over/Under execution
        if p_over > 0.54:
            if act_total == line_total:
                wagers.append({'Market': 'Total', 'Selection': 'Over', 'Win': 0.5, 'Profit': 0.0})
            else:
                win = 1.0 if act_total > line_total else 0.0
                profit = self.unit_size * self.standard_juiced_payout if win else -self.unit_size
                wagers.append({'Market': 'Total', 'Selection': 'Over', 'Win': win, 'Profit': profit})
        elif (1.0 - p_over) > 0.54:
            if act_total == line_total:
                wagers.append({'Market': 'Total', 'Selection': 'Under', 'Win': 0.5, 'Profit': 0.0})
            else:
                win = 1.0 if act_total < line_total else 0.0
                profit = self.unit_size * self.standard_juiced_payout if win else -self.unit_size
                wagers.append({'Market': 'Total', 'Selection': 'Under', 'Win': win, 'Profit': profit})
                
        return wagers
    def execute_historical_backtest(self):
        """Crawls local file paths directory logs to process full batch files settlement."""
        csv_files = glob.glob(self.predictions_path)
        
        # 🛡️ FIX: If no files exist yet, initialize an empty template file to prevent pipeline crashes
        if not csv_files:
            print("No daily historical prediction files located inside data/predictions/. Initializing blank ledger.")
            os.makedirs('data/analysis', exist_ok=True)
            pd.DataFrame(columns=['Market', 'Selection', 'Win', 'Profit']).to_csv('data/analysis/backtest_settlement_ledger.csv', index=False)
            return pd.DataFrame()
            
        compiled_wagers = []
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            date_str = file_name.replace("mlb_market_segments_", "").replace(".csv", "")
            
            actuals_db = self._fetch_actual_results(date_str)
            if not actuals_db:
                continue
                
            df_preds = pd.read_csv(file_path)
            fg_preds = df_preds[df_preds['Segment'] == 'Full Game']
            
            for _, row in fg_preds.iterrows():
                m_matchup = row['Matchup']
                if m_matchup in actuals_db:
                    game_bets = self._evaluate_bets(row, actuals_db[m_matchup])
                    compiled_wagers.extend(game_bets)
                    
        return pd.DataFrame(compiled_wagers)
    def display_roi_performance_dashboard(self, df_wagers):
        """Processes final transaction lists to generate exact performance sheets."""
        if df_wagers is None or df_wagers.empty:
            print("Insufficient settled wagers processed to output performance evaluations.")
            return
            
        total_wagers_placed = len(df_wagers)
        total_capital_risked = total_wagers_placed * self.unit_size
        net_profit_loss = df_wagers['Profit'].sum()
        
        win_pct = (df_wagers['Win'].sum() / total_wagers_placed) * 100.0
        final_roi = (net_profit_loss / total_capital_risked) * 100.0
        
        print("\n" + "="*45)
        print("   📈 QUANT ENGINE HISTORICAL ROI DASHBOARD   ")
        print("="*45)
        print(f" Total Settled Wagers Placed : {total_wagers_placed}")
        print(f" Total Capital Risked        : ${total_capital_risked:,.2f}")
        print(f" Net Model Profit / Loss     : ${net_profit_loss:,.2f}")
        print(f" Algorithmic Win Percentage  : {win_pct:.2f}%")
        print(f" Final Model Return (ROI%)   : {final_roi:.2f}%")
        print("="*45 + "\n")
        
        os.makedirs('data/analysis', exist_ok=True)
        df_wagers.to_csv('data/analysis/backtest_settlement_ledger.csv', index=False)
        print("Historical ledger ledger cleanly archived inside data/analysis/ path location.")

if __name__ == "__main__":
    backtester = MLBBacktestEngine()
    ledger_df = backtester.execute_historical_backtest()
    backtester.display_roi_performance_dashboard(ledger_df)
