"""Conditional normal-impact diagnostic using the v69 normal impulse oscillator."""
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys

DIRECTORY = Path(__file__).resolve().parent
# Cross (2014) author manuscript §2, Fig. 4b. Observation time starts at wire A.
RADIUS_M = 0.033
MEASURED_SPEED = 7.7                    # m/s, reported +/-0.1
MEASURED_DURATION = 0.00472             # s, reported +/-0.00001
RADII_MM = (5, 10, 15, 20)
OBSERVED_MS = (0.070, 0.218, 0.698, 1.408)
OBSERVED_DIAMETER_MM = 42.0              # reported +/-1


def predict(e, speed, duration):
    # The compression and restitution angular frequencies differ by factor e.
    # Each takes one quarter-cycle; durations add up to the measured tc.
    loading_frequency = math.pi * (1 + 1/e) / (2 * duration)
    indentation_max = speed / loading_frequency
    assert 0 < indentation_max < RADIUS_M
    radius_max = math.sqrt(2*RADIUS_M*indentation_max - indentation_max**2)
    times = []
    for r_mm in RADII_MM:
        r = r_mm / 1000
        indentation = RADIUS_M - math.sqrt(RADIUS_M**2 - r**2)
        if indentation > indentation_max:
            raise ValueError(f"Target radius {r_mm} mm is not reached for e={e}")
        time = math.asin(indentation/indentation_max) / loading_frequency
        check = speed * math.sin(loading_frequency*time) / loading_frequency
        assert math.isclose(math.sqrt(2*RADIUS_M*check - check**2), r, abs_tol=1e-12)
        times.append(1000*time)
    # Cross-check total impulse and total duration against solver.normal's convention.
    rebound_frequency = e*loading_frequency
    assert math.isclose(speed+rebound_frequency*indentation_max,(1+e)*speed)
    assert math.isclose(math.pi/2*(1/loading_frequency+1/rebound_frequency),duration)
    return times + [2000*radius_max]


def main():
    outcomes = [f"t_at_{r}mm" for r in RADII_MM] + ["max_diameter"]
    observations = list(OBSERVED_MS) + [OBSERVED_DIAMETER_MM]
    rows = []
    for e in (0.65, 0.75, 0.85):
        main_prediction = predict(e, MEASURED_SPEED, MEASURED_DURATION)
        observed_inputs = itertools.product((7.6, 7.8), (.00471, .00473))
        sensitivity = [predict(e,speed,duration) for speed,duration in observed_inputs]
        for i, (metric, value, prediction) in enumerate(zip(outcomes, observations, main_prediction)):
            rows.append(dict(restitution=e, metric=metric, unit="mm" if i==4 else "ms",
                             observed=value, predicted=prediction,
                             prediction_minus_observation=prediction-value,
                             predicted_low_input_sensitivity=min(c[i] for c in sensitivity),
                             predicted_high_input_sensitivity=max(c[i] for c in sensitivity)))
    with (DIRECTORY/'wire_grid_results.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    run = dict(host=platform.node(), python=sys.version, job=os.environ['SLURM_JOB_ID'],
               step=os.environ.get('SLURM_STEP_ID'), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
               input='Cross (2014), author manuscript section 2, Fig. 4(b), vertical Wilson Australian Open ball',
               source_url='https://doi.org/10.1007/s12283-014-0159-x',
               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DIRECTORY/'run.json').write_text(json.dumps(run,indent=2)+'\n')

if __name__ == '__main__':
    main()
