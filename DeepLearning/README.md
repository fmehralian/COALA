# Deep Learning Model 

To test a pretrained model on a sample data:
1. Download the test set and images (a sample is available in `out` dir)  
2. Download the model (a sample is available in `out/models` dir)
3. Download the vocab (a sample is available in `out/vocab_v1` dir)
4. Run the following command

`python3 start.py --test --image-path="out" --model-path=out/models/sample_trained --test-content-path="out/sample.json" --no-attention --log`


To train a new model from scratch:
1. Prepare the proper data annotation and split using `../DataAnalysis/split_data.py`
2. Adjust the config parameters on `start.py`, including vocab version.
3. Use the following command

`python3 start.py --image-path=<path to images> --test-content-path="<path to test.json>" --train-content-path="<path to train.json>" "`
