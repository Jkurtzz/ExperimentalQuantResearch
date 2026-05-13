from math import erfc, exp, sqrt
import logging
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2

log = logging.getLogger(__name__)

def apply_bh(group):
    _, p_bh, _, _ = multipletests(group['p_val'], method='fdr_bh')
    group['p_bh'] = p_bh
    return group

Z = 1.96
def wilson_ci(x, n, z=Z):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = x / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = (z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)) / denom
    return p, center - half, center + half

def newcombe_ci(x1,n1,x2,n2,z=Z):
    p1,L1,U1 = wilson_ci(x1,n1,z)
    p2,L2,U2 = wilson_ci(x2,n2,z)
    lift = p1 - p2 
    standard_err = (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
    return lift, standard_err, (L1 - U2), (U1 - L2)

def lor_and_se(a,b,c,d, cc=True):
    if cc:
        a,b,c,d = a+0.5, b+0.5, c+0.5, d+0.5
    lor = np.log((a*d)/(b*c))
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = lor/se
    return lor, se, z

def pre_press_hit_rate_comp(df, binary_feature, return_binary, min_spikes=50):
    try:
        test_data = df[[binary_feature, return_binary]]

        baseline_n = len(test_data[test_data[binary_feature] == 0])
        baseline_hits = test_data[test_data[binary_feature] == 0][return_binary].sum()
        baseline_no_hits = baseline_n - baseline_hits
        baseline_hit_rate = baseline_hits / baseline_n

        feature_n = len(test_data[test_data[binary_feature] == 1])
        if feature_n < min_spikes:
            log.warning(f"not enough data for {binary_feature}-{return_binary}")
            return

        feature_hits = test_data[test_data[binary_feature] == 1][return_binary].sum()
        feature_no_hits = feature_n - feature_hits
        feature_hit_rate = feature_hits / feature_n

        # if feature_hit_rate <= baseline_hit_rate + 0.05: 
        #     return  

        log.debug(f"{binary_feature}-{return_binary} baseline hit rate: {baseline_hit_rate} | feature hit rate: {feature_hit_rate} | baseline n: {baseline_n} | feature n: {feature_n}")

        lift, lift_se, newcombe_lower, newcombe_upper = newcombe_ci(x1=feature_hits, n1=feature_n, x2=baseline_hits, n2=baseline_n)
        log.debug(f"{binary_feature}-{return_binary} lift: {lift} | SE: {lift_se} | newcombe CI: [{newcombe_lower}, {newcombe_upper}]")

        lor, lor_se, z = lor_and_se(a=feature_hits, b=feature_no_hits, c=baseline_hits, d=baseline_no_hits)
        p_val = erfc(abs(z) / sqrt(2.0))
        log.debug(f"{binary_feature}-{return_binary} LOR: {lor} | SE: {lor_se} | z: {z} | p-value: {p_val}")

        # if newcombe_lower > 0:
        return {
            "dv_feature": binary_feature,
            "n_spikes": int(feature_n), 
            "n_baseline": baseline_n,
            "hit_rate": feature_hit_rate,
            "baseline_hit_rate": baseline_hit_rate, 
            "lift": lift,  
            "lift_se": lift_se, 
            "newcombe_lower": float(newcombe_lower), 
            "newcombe_upper": float(newcombe_upper), 
            "lor": lor, 
            "lor_se": lor_se, 
            "lor_ci_lower": lor - 1.96 * lor_se, 
            "lor_ci_upper": lor + 1.96 * lor_se, 
            "or": exp(lor),
            "z": z, 
            "p_val": p_val, 
            }
    except Exception as err:
        log.error(f"error analyzing hit rate: {err}", exc_info=True)


# entries: LORs or lifts
def heterogeneity(entries, ses):
    try:
        w = 1.0 / (ses ** 2)
        theta_fe = np.sum(w * entries) / np.sum(w) # fixed effect pooled logs odd ratio (LOR) or lift
        se_fe = (1.0 / np.sum(w)) ** 0.5 # standard error of theta_fe
        ci_lo, ci_hi = theta_fe - 1.96 * se_fe, theta_fe + 1.96 * se_fe # 95% CI for the pooled LOR or pooled lift
        or_fe = exp(theta_fe) # pooled odds ratio (OR)

        # test: do sectors differ in magnitude on the relative scale
        Q = np.sum(w * (entries - theta_fe) ** 2) # large q + small p = sector effects differ more than sampling noise can explain
        d_f = entries.size - 1
        p_Q = 1.0 - chi2.cdf(Q, d_f) 
        I2 = max(0.0, (Q - d_f) / Q) if Q > 0 else 0.0 # explains q as a percentage - x% is how much the spread is real sector-sector difference

        return theta_fe, or_fe, se_fe, (round(float(ci_lo), 5), round(float(ci_hi), 5)), Q, p_Q, I2
    except Exception as err:
        log.error(f"error computing heterogeneity: {err}", exc_info=True)