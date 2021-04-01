# About

This project is based on week 6 of Introduction to Machine Learning course available on Coursera.
(https://www.coursera.org/learn/intro-to-deep-learning/)


## Training

The resources for training can be downloaded from course materials at
(https://github.com/hse-aml/intro-to-dl/releases/tag/v0.1)
```
[
    "captions_train-val2014.zip",
    "train2014_sample.zip",
    "train_img_embeds.pickle",
    "train_img_fns.pickle",
    "val2014_sample.zip",
    "val_img_embeds.pickle",
    "val_img_fns.pickle"
]
```

Put them anywhere you like and update the "fix_path" function in utils.py

Then execute `python run-keras.py -m t` to start training.


## Prediction

Either do your own training or download the pre-trained checkpoints at
(https://github.com/yisyang/coursera-ml-course-week6/releases/tag/v1.1)

Then put your example images into `./examples`.

Execute `python run-keras.py -m p -l 24` to start training,
where 24 is the last saved checkpoint under `./checkpoints`.


## Examples
![results](https://user-images.githubusercontent.com/5167456/113249749-1a4c4600-9274-11eb-9f66-07d03bae03a6.png)
