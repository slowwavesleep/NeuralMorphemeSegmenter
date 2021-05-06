# NeuralMorphemeSegmenter

## Run experiment
```
python run_experiment.py  -p configs/lstmcrf_lemmas.yml
```

## Run experiment job

```
bash run_experiment_job.sh "./experiment_jobs/lstm" 3 3
```
## Generate predictions on test

```
python run_test_segmentation.py --model_name LstmTagger --train_type forms
```
## Experiment info

### Number of runs for each model/data type combination


|Model|Lemmas|Lemmas Low Resource|Forms|Forms Low Resource|Forms Shuffled|Forms Shuffled Low Resource|
|:---|:---:|:---:|:---:|:---:|:---:|---:|
|Random`*`|0|0|0|0|0|0|
|Baseline|1|1|1|1|1|1|
|Baseline-CRF|1|1|1|1|1|1|
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
#### Ratio of fully correct predictions
|Model|Lemmas|Lemmas Low Resource|Forms|Forms Low Resource|Forms Shuffled|Forms Shuffled Low Resource|
|:---|:---:|:---:|:---:|:---:|:---:|---:|
|Random|0.000069|-|0.000079|-|0.000178|-|
|Baseline|0.0032|0.003|0.0018|0.0017|0.0019|0.0019|
|Baseline-CRF|0.18|0.1553|0.1858|0.1652|0.1853|0.1907|
|LSTM|0.898|0.6312|0.7567|0.4962|0.9936|0.777|
|LSTM-CRF|0.9059|0.6464|0|0|0|0|
|CNN|0.7255|0.4168|0|0|0|0|
|CNN-CRF|0.4375|0|0|0|0|0|
|Transformer|0|0|0|0|0|0|
|Transformer-CRF|0|0|0|0|0|0|

_Low resource settings use the same valid/test data as their full counterparts_ 