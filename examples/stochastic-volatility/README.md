## Threshold Analysis in the Stochastic Volatility Model

To generate observations from the stochastic volatility posterior distribution distribution, execute:
```
python generate_data.py
```
To visualize the transition kernel distance, execute:
```
sbatch transition_kernel_distance.sh
```
Examine the performance of the generalized leapfrog integrator with varying thresholds.
```
python generate_joblist.py
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 2 -t 24:00:00 --job-name stochastic-volatility -o output/stochastic-volatility-%A-%J.log --suppress-stats-file --submit
```
