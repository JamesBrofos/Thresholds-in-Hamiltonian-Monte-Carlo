## Threshold Analysis in Student's t-Distribution

To generate baseline data from the multivariate Student's t-distribution, execute:
```
sbatch generate_samples.sh
python unit_vectors.py
```
To visualize the adaptation using Ruppert averaging, execute:
```
sbatch adaptation.sh
```
Examine the performance of the generalized leapfrog integrator with varying thresholds.
```
python generate_joblist.py
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 1 -t 24:00:00 --job-name student -o output/student-%A-%J.log --suppress-stats-file --submit
```
To compute metrics based on the samples generated by these Monte Carlo methods, execute:
```
python generate_metricslist.py 
dSQ -C cascadelake --jobfile metricslist.txt -p week --max-jobs 1000 -c 1 -t 168:00:00 --job-name student-metrics -o output/student-metrics-%A-%J.log --suppress-stats-file --submit
```
