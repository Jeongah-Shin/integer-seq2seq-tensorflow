# integer-seq2seq-tensorflow

### Problem

> The objective of this task is to create a model that approximates a mapping function from an input sequence of integers ("source") to an output sequence of integers ("target") using a training data set (`train_source.txt`, `train_target.txt`), and achieve best generalization performance on a held-out test set (`test_source.txt`, `test_target.txt`). Use any technique and framework you think is appropriate.
>
> Please submit a link to a GitHub repository, containing your code, and `README` which describes the following:
>
> 1. your experiment design (including baselines and models and/or data exploration results)
> 2. evaluation metrics
> 3. experimental results.



### Settings

For activate the virtual environment for testing model,

```bash
source ./arie-tf-2.0/bin/activate
```

***or***

For your custom virtual environment,

```bash
pip3 install -r requirements.txt
```



### Data Exploration

|                                 | Train set | Test set |
| ------------------------------- | --------- | -------- |
| Number of samples               | 7260      | 2000     |
| Number of unique input tokens   | 11        | 11       |
| Number of unique output tokens  | 13        | 13       |
| Max sequence length for inputs  | 303       | 303      |
| Max sequence length for outputs | 197       | 197      |



### Experimental Design

<p align="center">
  		<img src ="./images/0_experiment_design.png?raw=true" height="200px"/>
</p>

  The first thing to do was converting input sequence to states vector. Next, produce next prediction of integer by inserting the states vectors and target sequence(begins with target sequence with size 1 - starting annotation character *'s'*). For inferencing the model, choose the sample utilizing these predictions(using `argmax`) and add the sampled character(in this case **integer**) into result. Repeat all the processes when meets the sequence end character(*'e'*).




### Cost Function / Optimizer

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

| Options                 | Selection                      |
| ----------------------- | ------------------------------ |
| Optimizer               | RMSProp                        |
| Loss calculation method | Categorial CrossEntropy        |
| Evaluation Metric       | Accuracy (Categorial Accuracy) |



### Training
<p align="center">
      <img src ="./images/2_training_acc.png?raw=true" height="250px"/>
  		<img src ="./images/3_training_loss.png?raw=true" height="250px"/>
</p>

#### T1

```json
  "baseline_with_val_0.2" :
  {
    "batch_size" : 32,
    "epochs" : 100,
    "latent_dim" : 40,
    "num_samples" : 7260,
    "num_encoder_tokens" : 11,
    "num_decoder_tokens" : 13,
    "validation_split" : 0.2
  }
```
#### T2

```json
  "b64_no_val" :
  {
    "batch_size" : 64,
    "epochs" : 100,
    "latent_dim" : 40,
    "num_samples" : 7260,
    "num_encoder_tokens" : 11,
    "num_decoder_tokens" : 13,
    "validation_split" : 0.0
  }
```
#### T3

```json
  "b64_with_val" :
  {
    "batch_size" : 64,
    "epochs" : 100,
    "latent_dim" : 40,
    "num_samples" : 7260,
    "num_encoder_tokens" : 11,
    "num_decoder_tokens" : 13,
    "validation_split" : 0.2
  }
```
Total parameters : 17,493

|          | T1         | T2     | T3     |
| -------- | ---------- | ------ | ------ |
| Loss     | 0.1518     | 0.1559 | 0.1595 |
| Accuracy | **0.9483** | 0.9467 | 0.9457 |



### Experiment Result

- On Test Set (2000 Lines)

|                   | Test Loss               | Test Accuracy (Categorial) |
| ----------------- | ----------------------- | -------------------------- |
| T1 **(baseline)** | **0.15920982682704926** | **0.9459543**              |
| T2                | 0.16035067236423492     | 0.9456472                  |
| T3                | 0.16392553293704987     | 0.9444162                  |



### Retrospection

- Learning curves

    Some difficulites exist since it is the first time to me to design and implement RNN models for my own. The following two points of view are regretful things to me that I want to make up in the future.

  **Further Approach**

    - Code Convention
    - Further Approach - Attention mechanism, BLEU Score

- NMT(Neural Machine Translation) as On-Device ML

    As a perspective of Light-weight DNN, I think it is important to minimize pre-processing/post-processing  since the heavy operations on mobile or edge devices should be avoided. Therfore, It would be preferable **to carry out the processes**(`process_data.py`) **within the model** with this perspective. Also, using **GRU-based encoder/decoder** would be preferable instead of LSTM that it uses less training(or total) parameters and less memories.

Thank you for the good opportunity! It was a task that left a lot of regrets while having fun with new experiences. Maybe It sounds like an excuse for myself, I want to try again after having enough time to have a deep understanding of the NMT(Neural Machine Translation).



### TODO 

- [x] virtual environment settings for implementing tensorflow 2.0
- [x] pylint settings for IDE
  - reference : https://code-examples.net/en/q/245e146
- [x] problem definition 
- [x] experiment design
  - [x] baselines
  - [x] models
  - [x] data exploration results
- [x] cost function / optimizer selection
- [x] train model
	- [x] data feeding - batch size
	- [x] optimization solver - training epoch, learning rate
- [x] evaluation metrics
- [x] experimental results



### Challenges

- [ ] Tensorflow Lite Conversion (if possible in time)
- [ ] Tensorflow Lite Testing (if possible in time)

> ***19.10.30 updated***
>
> LSTM Operations are present but not fully ready for custom models. However, it could be dealed with `SELLECT_OPS` system via Tensorflow Lite.