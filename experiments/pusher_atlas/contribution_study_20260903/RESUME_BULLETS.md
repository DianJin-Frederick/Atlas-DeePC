# Verified résumé wording — development stage only

> Archived pre-final wording. Use [FINAL_RESUME_BULLETS.md](FINAL_RESUME_BULLETS.md)
> for claims based on the now-completed 2,000-rollout evaluation.

Recommendation: use **one modest supporting bullet**, not a standalone claim
that hybrid Atlas-DeePC outperforms imitation. The new 100-seed paired final
evaluation has not run. These are alternative candidates, not three separate
research achievements to pad the résumé.

1. **Built a reproducible comparison of imitation-only, predictive, and hybrid
   Atlas-DeePC controllers, measured by 50 completed development rollouts on
   two paired Pusher-v5 seeds, by separating policy-guided retrieval from action
   regularization and logging computation, outcomes, and constraint excursions.**
   Scope: five methods, nominal/sliding-resistance/mass conditions and two fixed
   disturbance timings; not 50 independent seeds or held-out trials.
   [40-trial record](pilot/per_trial.csv), [10-trial recovery record](recovery_pilot/summary.json),
   [baseline definitions](README.md).

2. **Established a matched imitation-only baseline against policy-guided
   Atlas-DeePC, measured by zero decoder error in 200 expert/state checks and
   exact combined-controller parity on one development seed, by reusing the
   joint-target-to-PD torque interface and validating the evaluation harness.**
   Scope: numerical implementation validation, not superior control performance.
   [Decoder checks](audit.json), [rollout parity](validation.json).

3. **Verified a historical 90/100 versus 0/100 sustained-success gap between
   policy-guided and strict Atlas-DeePC on Pusher seeds 201–300, measured by
   recomputation of 200 saved trajectories, by auditing the final-eight-step
   criterion and identifying confounded guidance ablations.**
   Scope: saved-data verification, not a new execution or evidence that DeePC
   improves imitation. Historical action replay diverges under the current
   engine, so prefer candidate 1 or 2 until reproducibility is resolved.
   [Historical audit](audit.json), [limitations](CONCLUSION.md).

Do not claim improved robustness, recovery, safety, real-time deployment,
sim-to-real transfer, or a 90-percentage-point contribution from DeePC based
on the completed evidence. Resume claims should be revised only after the
fresh frozen evaluation is completed and interpreted.
