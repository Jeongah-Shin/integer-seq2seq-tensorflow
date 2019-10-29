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

"눈으로 이해 가능하고 손으로 조작 가능한 입출력 범위 찾기"



### TODO 

- [x] virtual environment settings for implementing tensorflow 2.0
- [x] pylint settings for IDE
  - reference : https://code-examples.net/en/q/245e146
- [ ] problem definition 
- [ ] EDA
- [ ] experiment design
  - [ ] baselines
  - [ ] models
  - [ ] data exploration results
- [ ] cost function / optimizer selection
- [ ] train model
	- [ ] data feeding - batch size
	- [ ] optimization solver - training epoch, learning rate
- [ ] evaluation metrics
- [ ] experimental results



### Challenges

- [ ] Tensorflow Lite Conversion (if possible in time)
- [ ] Tensorflow Lite Testing (if possible in time)