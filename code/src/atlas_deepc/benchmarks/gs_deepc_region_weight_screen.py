"""Small follow-up screen using only existing GS-DeePC tuning parameters."""
from atlas_deepc.benchmarks.gs_deepc_region_tuning import run

# The previous q2-tuned weights need not suit link2 scheduling. Compare matched
# seeds and retain all unsuccessful trials. No automatic search or early stop.
CASES = [
    ('link2_regions4_r05',4,.5,10.),
    ('link2_regions8_r05',8,.5,10.),
    ('link2_regions4_r005',4,.05,10.),
    ('link2_regions8_r005',8,.05,10.),
    ('link2_regions4_r05_g100',4,.5,100.),
    ('link2_regions8_r05_g100',8,.5,100.),
]


if __name__=='__main__':
    for name,regions,r,g in CASES:
        run(name,'link2',regions,[211,222],r=r,lambda_g=g)
