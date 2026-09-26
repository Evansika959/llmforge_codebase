# Recorded device-state ablation

Offline XGBoost comparison on the historical frozen 1,564 architectures, using
the original 1,100/150/314 split and three training seeds. Both architecture14
and physics32 are compared with and without recorded starting temperature.
Hardware profiles are fitted on training rows only; validation selects stopping
iterations. The architecture14 control must reproduce historical test metrics.

```bash
python -m scripts.prediction_research.state_inputs.run \
  --output scripts/prediction/outputs/state_inputs_1564
```

Choose a new output directory for each run. Inputs require the locally retained
`outputs/batch2_progress_632` snapshots. Outputs include a manifest, twelve
diagnostic checkpoints, test predictions, summary metrics and a report.

Checkpoints are diagnostic dictionaries, **not production `predict_bundle`
bundles**. Build the feature matrix in the stored `features` order (using the
stored physics profile where present and appending recorded temperature), then
call `inference.trees.tree_predict(models, x, inverse_tpot)`. Only load trusted
joblib files. These experiments do not access hardware or deploy AL models.

Initial battery percentage and initial voltage are absent from this frozen
cohort. `voltage_v` is post-run and excluded. Temperature was sampled before the
run, not verified immediately before inference. This is device-state-conditioned
prediction, not architecture-only search. The reused test split makes results
exploratory; correlations do not establish causal temperature effects.

The completed three-seed comparison gives architecture14 MAPE of
20.80/19.06/26.69% (throughput/TTFT/dynamic energy), versus
10.38/8.90/18.90% with starting temperature. Physics32 with temperature gives
10.60/8.87/18.46%. See the output report for uncertainty and subgroup results.
