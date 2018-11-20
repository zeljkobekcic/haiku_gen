# Haiku Gen(erator)

This is a neural network which is trained to generate haikus. Currently there is only a character based prediction but
there will be a syllable and word based prediction too.

# Want to train it yourself?

You need to create a *virtualenv* and activate it. Furthermore you need to have Python3 and *pip3* installed.

First you need to get the data, just follow the readme in 
[haiku_scrapper](https://github.com/zeljkobekcic/haiku_scrapper).
You will need this data for the preprocessing. Just remember where you store it.


```bash
git clone git@github.com:zeljkobekcic/haiku_gen.git
cd haiku_gen
bash setup.bash
pip install -r requirements.txt
ipython haiku_gen/preprocessing/haiku_preprocessing.py -- --data data --input <path_to_the_raw_data>
```

Then you can train the model.

```bash
ipython haiku_gen/nn.py
```

The model will be saved in the `model` directory.