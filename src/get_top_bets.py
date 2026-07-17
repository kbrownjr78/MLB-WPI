import os
import datetime
import pandas as pd

def extract_top_five_wagers():
    today = datetime.date.today().strftime('%Y-%m-%d')
    segments_path = f"data/predictions/mlb_market_segments_{today}.csv"
    props_path = f"data/predictions/mlb_dk_props_{today}.csv"
    
    # 1. Verification Check: Ensure files exist for the current date
    if not os.path.exists(segments_path) or not os.path.exists(props_path):
        print(f"⚠️ Predictions for {today} not found. Run 'python src/predict_games.py' first.")
        return

    # 2. Read Daily CSV Outputs
    df_seg = pd.read_csv(segments_path)
    df_prop = pd.read_csv(props_path)
    
    compiled_opportunities = []

    # 3. Parse Segment Markets (Moneylines and Totals)
    for _, row in df_seg.iterrows():
        # Strip '%' tags and cast to float percentages
        h_ml = float(row['Home_ML_Probability'].replace('%', ''))
        a_ml = float(row['Away_ML_Probability'].replace('%', ''))
        over_p = float(row['Over_Total_Probability'].replace('%', ''))
        under_p = float(row['Under_Total_Probability'].replace('%', ''))
        
        # Isolate the highest-probability outcome for the row
        if h_ml > a_ml:
            compiled_opportunities.append({'Target': row['Matchup'], 'Market': f"{row['Segment']} Moneyline", 'Selection': 'Home Team', 'Probability': h_ml})
        else:
            compiled_opportunities.append({'Target': row['Matchup'], 'Market': f"{row['Segment']} Moneyline", 'Selection': 'Away Team', 'Probability': a_ml})
            
        if over_p > under_p:
            compiled_opportunities.append({'Target': row['Matchup'], 'Market': f"{row['Segment']} Total", 'Selection': f"OVER {row['Target_DK_Total_Line']}", 'Probability': over_p})
        else:
            compiled_opportunities.append({'Target': row['Matchup'], 'Market': f"{row['Segment']} Total", 'Selection': f"UNDER {row['Target_DK_Total_Line']}", 'Probability': under_p})

    # 4. Parse Player Props (Strikeouts, Hits, Total Bases, etc.)
    for _, row in df_prop.iterrows():
        over_p = float(row['Over_Probability'].replace('%', ''))
        under_p = float(row['Under_Probability'].replace('%', ''))
        
        if over_p > under_p:
            compiled_opportunities.append({'Target': row['Player_Name'], 'Market': row['Market_Type'], 'Selection': f"OVER {row['Most_Likely_Line_Outcome']}", 'Probability': over_p})
        else:
            compiled_opportunities.append({'Target': row['Player_Name'], 'Market': row['Market_Type'], 'Selection': f"UNDER {row['Most_Likely_Line_Outcome']}", 'Probability': under_p})

    # 5. Sort by Highest Absolute Probability and Extract Top 5
    df_all = pd.DataFrame(compiled_opportunities)
    df_top_5 = df_all.sort_values(by='Probability', ascending=False).head(5).reset_index(drop=True)
    
    # Print the Results Card
    print("\n" + "🥇" + "="*50 + "🥇")
    print(f"     🔥 MODEL TOP 5 HIGHEST PROBABILITY BETS FOR {today} 🔥     ")
    print("="*54)
    for idx, row in df_top_5.iterrows():
        print(f" {idx+1}. {row['Target']} | {row['Market']}")
        print(f"    🎯 SELECTION: {row['Selection']} --> PROBABILITY: {row['Probability']:.1f}%\n")
    print("="*54 + "\n")

if __name__ == "__main__":
    extract_top_five_wagers()
