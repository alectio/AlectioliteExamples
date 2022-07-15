# Topic Classification on Reuters-21578 News Dataset

This example is intended to show you how to build the `train`, `test` and `infer` processes for the `AlectioSDK` for topic
classification problems. We will use the [Reuters](https://martin-thoma.com/nlp-reuters/) dataset. We focus on the 20 most popular news topics. An example sentence is


| label | topic |
| ----- | ----- |
| interest   | FED EXPECTED TO ADD RESERVES The Federal Reserve is expected to enter the U.S. government securities market to add reserves during its usual intervention period today, economists said. With federal funds trading at a steady 6-3/16 pct, most economists expect an indirect injection of temporary reserves via a medium-sized round of customer repurchase agreements. However, some economists said the Fed may arrange more aggressive system repurchase agreements. Economists would also not rule out an outright bill pass early this afternoon. Such action had been widely anticipated yesterday but failed to materialize.


### 1. Set up a virtual environment and install Alectiolite SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment. 

To recap, the steps were setting up a virtual environment and then installing the Alectiolite SDK in that environment. 

To install the AlectioSDK from within the current directory (`./examples/NLP_classification`) run:

```
pip install ../../.
```

Next, go to the Alectio Frontend and download the API key by hitting the `DOWNLOAD API KEY` button. This will download a file called `credentials.json` which you should place in the current working directory (`./examples/NLP_classification`).

### 2. Get Code, Data and Dependencies 

First, point your terminal to the directory of this Readme file. Your terminal should look like this:
```bash 
(env)$~/AlectioliteExamples/NLP_classification
```
Then, clone the `hedwig` repo, which contains our main model
```shell
git clone https://github.com/castorini/hedwig
```

For the latest changes (which are still not merged), fetch the pull request

```shell
git fetch origin pull/81/head:upgraded_for_transformers-4.19
git checkout upgraded_for_transformers-4.19
```

If successful, you should have a folder within your SDK repo called `hedwig`. It should look like this:

```
├── examples
│   ├── NLP_classification
│   │   └── reuters_hedwig
│   ├── image_classification
│   ├── object_detection
│   └── topic_classification
```

Then install pytorch with

```
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

```

then the remaining requirements. some of the dependencies are old, so lets make sure we install the latest-ones

```shell
sed -i -e 's/transformers==2.1.1/transformers/g' requirements.txt
sed -i -e 's/numpy.*/numpy/g' requirements.txt
sed -i -e 's/scikit-learn.*/scikit-learn/g' requirements.txt
sed -i -e 's/scipy.*/scipy/g' requirements.txt
````

```
pip install -r requirements.txt
```

Now download the Reuters, AAPD, and IMDB datasets, along with word2vec embeddings.
```
git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

Cloning can take lot of time, so we will take a shortcut. We will download zip file from google-drive and unzip it. 

install gdown to download zip file from google-drive

```shell
cd ..
gdown 1qHgNuSMbDa8qETW_6SAKpLEQQG3JQY8H
unzip hedwig-data.zip #unzip the downloaded file
```

Incase, you want to run and verify model without Alectio-sdk

```shell
cd hedwig
python -m models.bert --dataset Reuters --model bert-base-uncased \  --max-seq-length 256 --batch-size 16 --lr 2e-5 --epochs 30
```

Now, reuters repo expects hedwig-data in parent folder, so create a symlink to hedwig-data from parent folder

```shell
cd /content/AlectioliteExamples
ln -s NLP_classification/hedwig-data hedwig-data
cd /content/AlectioliteExamples/NLP_classification
```

### 4. Build Model
We use the BERT topic classification model that is already implemented in the reuters-hedwig repo. 


### 3. Build Test Process
The test process tests the model trained in each active learning loop.
In this example, the test process is the `test` function defined 
in `processes.py`. 

```
python processes.py
```

#### Return of the Test Process 
You will need to run non-maximum suppression on the predictions on the test images and return 
the final detections along with the ground-truth bounding boxes and objects
on each image. 

The return of the `test` function is a dictionary 
```
{"predictions": predictions, "labels": labels}
```

Both `predictions` and `labels` are lists where `labels` denotes the ground truths for the images.

### 4. Build Infer Process
The infer process is used to apply the model to the unlabeled set to run inference. 
We will use the inferred output to estimate which of those unlabeled data will
be most valuable to your model.

Refer to main [AlectioSDK ReadMe](../../README.md) for general information regarding the 
arguments of this process.

#### Return of the Infer Process
The return of the infer process is a dictionary
```python
{"outputs": outputs}
```

`outputs` is a dictionary whose keys are the indices of the unlabeled
images. The value of `outputs[i]` is a dictionary that records the output of
the model on training image `i`. 

### 5. Build Flask App 
First you have to set up your main.py file to contain the token that is specific to your experiment. After you
create an experiment on the Alectio platform, you will receive a unique token that will be necessary to run your experiment.

Copy and paste that token into the main.py file under the token field within the Pipeline object.
```python
app = Pipeline(
    name="cifar10",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    token='<your-token-here>'
)
```
Once you have updated that file, execute the python file and you should be able to begin running your experiment.
```shell script
python main.py
```


