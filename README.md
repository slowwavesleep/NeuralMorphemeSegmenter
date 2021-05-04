# NeuralMorphemeSegmenter

## Run experiment
```
python run_experiment.py  -p configs/lstmcrf_lemmas.yml
```

## Run experiment job

```
bash run_experiment_job.sh "./experiment_jobs/lstm" 3 3
```

## Experiment info

### Number of runs for each model/data type combination


|Model|Lemmas|Lemmas Low Resource|Forms|Forms Low Resource|Forms Shuffled|Forms Shuffled Low Resource|
|:---|:---:|:---:|:---:|:---:|:---:|---:|
|Random`*`|0|0|0|0|0|0|
|Baseline|1|0|1|0|1|0|
|Baseline-CRF|1|0|1|0|1|0|
|LSTM|3|3|3|3|3|3|
|LSTM-CRF|3|3|3|3|3|3|
|CNN|3|3|3|3|3|3|
|CNN-CRF|3|3|3|3|3|3|
|Transformer|3|3|3|3|3|3|
|Transformer-CRF|3|3|3|3|3|3|

All`*` models are trained for 100 epochs with early stopping if there's no improvement
in the ratio of correctly predicted examples for 10 epochs in a row.

`*` Random model is not trained

### Best scores on test for each model/data type combination

|Model|Lemmas|Lemmas Low Resource|Forms|Forms Low Resource|Forms Shuffled|Forms Shuffled Low Resource|
|:---|:---:|:---:|:---:|:---:|:---:|---:|
|Random`*`|0|0|0|0|0|0|
|Baseline|0|0|0|0|0|0|
|Baseline-CRF|0|0|0|0|0|0|
|LSTM|0|0|0|0|0|0|
|LSTM-CRF|0|0|0|0|0|0|
|CNN|0|0|0|0|0|0|
|CNN-CRF|0|0|0|0|0|0|
|Transformer|0|0|0|0|0|0|
|Transformer-CRF|0|0|0|0|0|0|

_Low resource settings use the same valid/test data as their full counterparts_ 