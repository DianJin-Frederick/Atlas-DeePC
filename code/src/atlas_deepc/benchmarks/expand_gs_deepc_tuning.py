"""Second, explicit weight-only GS-DeePC pilot batch (no formulation changes)."""
from atlas_deepc.benchmarks.tune_gain_scheduled_deepc import run

# Fix Q=100: only relative objective weights affect the mathematical optimizer.
# Explore regions omitted by the first pilot, particularly small R and lambda_g.
# Tuple order: name, R, lambda_g, lambda_ini. No outcomes used to choose seeds.
CASES = [
    ('lowr_g1000', 1e-4, 1e3, 1e6),
    ('lowr_g10', 1e-4, 10., 1e6),
    ('lowr_g01', 1e-4, .1, 1e6),
    ('lowr_g0001', 1e-4, .001, 1e6),
    ('r001_g1', .01, 1., 1e6),
    ('r001_g100', .01, 100., 1e6),
    ('r001_g1e4', .01, 1e4, 1e6),
    ('ini100_r001_g10', .01, 10., 100.),
    ('ini1e4_r001_g10', .01, 10., 1e4),
    ('ini1e4_r01_g1', .1, 1., 1e4),
    ('ini1e4_r1_g100', 1., 100., 1e4),
    ('ini100_r001_g001', .01, .01, 100.),
    ('ini1e8_lowr_g001', 1e-4, .01, 1e8),
    ('ini1e8_lowr_g1', 1e-4, 1., 1e8),
    ('ini1e8_r001_g100', .01, 100., 1e8),
    ('ini1e8_r01_g10', .1, 10., 1e8),
    ('r01_g1', .1, 1., 1e6),
    ('r01_g10', .1, 10., 1e6),
]


if __name__ == '__main__':
    for name, r, g, ini in CASES:
        run('expand_'+name, [211, 222], 300, 100., r, g, ini)
