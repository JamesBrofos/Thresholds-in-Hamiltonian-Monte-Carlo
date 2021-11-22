## Threshold Analysis in the Cox-Poisson Model

To generate observations from the log-Gaussian Cox-Poisson posterior distribution distribution, execute:
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
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 2 -t 24:00:00 --job-name cox-poisson -o output/cox-poisson-%A-%J.log --suppress-stats-file --submit
```
