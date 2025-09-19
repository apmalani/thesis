import numpy as np
import pandas as pd

class GerrymanderingRewardFunction:
    def __init__(self, baseline_stats_file):
        self.baseline_stats = pd.read_csv(baseline_stats_file, index_col=0)
    
    def standardize_metric(self, metric_name, value, use_median=True):
        stats = self.baseline_stats.loc[metric_name]
        center = stats['median'] if use_median else stats['mean']
        scale = stats['std']
        
        return (value - center) / scale
    
    def calculate_reward(self, district_metrics, weights=None, clip_extreme=True):
        if weights is None:
            weights = {
                'EfficiencyGap': 1.0,
                'PartisanProp': 1.0, 
                'SeatsVotesDiff': 1.0,
                'PolPopperMin': 1.0,
                'PolPopperAvg': 1.0,
                'MinOppMin': 1.0,
                'MinOppAvg': 1.0
            }
        
        EG_z = self.standardize_metric('EfficiencyGap', district_metrics['EfficiencyGap'])
        PP_z = self.standardize_metric('PartisanProp', district_metrics['PartisanProp'])
        SV_z = self.standardize_metric('SeatsVotesDiff', district_metrics['SeatsVotesDiff'])
        MP_z = self.standardize_metric('PolPopperMin', district_metrics['PolPopperMin'])
        AP_z = self.standardize_metric('PolPopperAvg', district_metrics['PolPopperAvg'])
        MD_z = self.standardize_metric('MinOppMin', district_metrics['MinOppMin'])
        AD_z = self.standardize_metric('MinOppAvg', district_metrics['MinOppAvg'])
        
        if clip_extreme:
            EG_z = np.clip(EG_z, -3, 3)
            PP_z = np.clip(PP_z, -3, 3)
            SV_z = np.clip(SV_z, -3, 3)
            MP_z = np.clip(MP_z, -3, 3)
            AP_z = np.clip(AP_z, -3, 3)
            MD_z = np.clip(MD_z, -3, 3)
            AD_z = np.clip(AD_z, -3, 3)
        
        R = -(weights['EfficiencyGap'] * EG_z + 
              weights['PartisanProp'] * PP_z + 
              weights['SeatsVotesDiff'] * SV_z) + \
            weights['PolPopperMin'] * MP_z + \
            weights['PolPopperAvg'] * AP_z + \
            weights['MinOppMin'] * MD_z + \
            weights['MinOppAvg'] * AD_z
        
        return R
    