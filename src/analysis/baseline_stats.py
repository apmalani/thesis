import pandas as pd
import numpy as np

def calculate_baseline_stats(state, basepath):
        df = pd.read_csv(f"{basepath}/{state}/ensemble_metrics.csv")
        metrics = ['EfficiencyGap', 'PartisanProp', 'SeatsVotesDiff', 
                       'PolPopperAvg', 'PolPopperMin', 'MinOppAvg', 'MinOppMin']
        baseline_stats = {}

        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                baseline_stats[metric] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max(),
                    'count': len(values)
                }
    
        baseline_df = pd.DataFrame(baseline_stats).T
        baseline_df.to_csv(f"{basepath}/{state}/baseline_stats.csv")

        return baseline_df