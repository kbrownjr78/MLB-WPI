import os
import glob
import datetime
import numpy as np
import pandas as pd
import statsapi

class MLBBacktestEngine:
    def __init__(self, unit_size=100.0):
        # Establish testing bankroll constraints
        self.unit_size = unit_size
        self.predictions_path = "data/predictions/mlb_market_segments_*.csv"
        
        # Standardize standard bookmaker vigorish juiced odds fallbacks
        # -110 means wager $110 to win $100 profit
        self.standard_juiced_payout = 100.0 / 110.0 
        # +100 even money proxy fallback for standard moneylines
        self.moneyline_payout_multiplier = 1.00 
    def _fetch_actual_results(self, target_date):
        """Queries MLB StatsAPI to grab official game scores for verification."""
        print(f"Retrieving official structural box scores for {target_date}...")
        results = {}
        try:
            # Pull full slate matching historical date parameter
            games = statsapi.schedule(date=target_date)
            for g in games:
                if g.get('status') != 'Final':
                    continue
                    
                # Clean team name keys to ensure exact matching loops
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
        """Applies true sports betting settlement matrix constraints to calculate profit/loss."""
        p_home_ml = float(pred_row['Home_ML_Probability'].replace('%', '')) / 100.0
        p_away_ml = float(pred_row['Away_ML_Probability'].replace('%', '')) / 100.0
        p_over = float(pred_row['Over_Total_Probability'].replace('%', '')) / 100.0
        
        line_total = float(pred_row['Target_DK_Total_Line'])
        
        # Pull actual final scores
        act_away = actual['away_score']
        act_home = actual['home_score']
        act_total = actual['total_runs']
        
        wagers = []
        
        # 1. SETTLE MONEYLINE MARKETS (+EV Threshold configured at 54% Probability Edge)
        if p_home_ml > 0.54:
            win = 1.0 if act_home > act_away else 0.0
            profit = self.unit_size * self.moneyline_payout_multiplier if win else -self.unit_size
            wagers.append({'Market': 'Moneyline', 'Selection': 'Home', 'Win': win, 'Profit': profit})
        elif p_away_ml > 0.54:
            win = 1.0 if act_away > act_home else 0.0
            profit = self.unit_size * self.moneyline_payout_multiplier if win else -self.unit_size
            wagers.append({'Market': 'Moneyline', 'Selection': 'Away', 'Win': win, 'Profit': profit})
            
        # 2. SETTLE TOTALS OVER/UNDER MARKETS
        if p_over > 0.54:
            if act_total == line_total: # Push condition handler
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
        if not csv_files:
            print("No daily historical prediction files located inside data/predictions/ path targets.")
            return
            
        compiled_wagers = []
        
        for file_path in csv_files:
            # Extract date timestamp token structure from layout file name layout string
            file_name = os.path.basename(file_path)
            date_str = file_name.replace("mlb_market_segments_", "").replace(".csv", "")
            
            # Extract official scores matching timestamp sequence loops
            actuals_db = self._fetch_actual_results(date_str)
            if not actuals_db:
                continue
                
            df_preds = pd.read_csv(file_path)
            # Filter solely down to Full Game profiles for historical validation integrity
            fg_preds = df_preds[df_preds['Segment'] == 'Full Game']
            
            for _, row in fg_preds.iterrows():
                m_matchup = row['Matchup']
                if m_matchup in actuals_db:
                    game_bets = self._evaluate_bets(row, actuals_db[m_matchup])
                    compiled_wagers.extend(game_bets)
                    
        return pd.DataFrame(compiled_wagers)
    def display_roi_performance_dashboard(self, df_wagers):
        """Processes final transaction lists to generate exact performance sheets."""
        if df_wagers is empty if 'df_wagers' not in locals() else df_wagers.empty:
            print("Insufficient settled wagers processed to output performance evaluations.")
            return
            
        total_wagers_placed = len(df_wagers)
        total_capital_risked = total_wagers_placed * self.unit_size
        net_profit_loss = df_wagers['Profit'].sum()
        
        # Calculate win efficiency percentages (counting pushes cleanly as half-wins)
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
        
        # Archive results data tracking metrics file to branch path
        os.makedirs('data/analysis', exist_ok=True)
        df_wagers.to_csv('data/analysis/backtest_settlement_ledger.csv', index=False)
        print("Historical ledger ledger cleanly archived inside data/analysis/ path location.")

if __name__ == "__main__":
    backtester = MLBBacktestEngine()
    ledger_df = backtester.execute_historical_backtest()
    backtester.display_roi_performance_dashboard(ledger_df)
