# Deep Learning Model 

To test a pretrained model on a sample data:
- Extract the content of sample model and data vocab in `out` directory [drive](https://drive.google.com/file/d/1Va2YKgK_7K1KQ9hXAgG6vnKs35Vc-mcQ/view?usp=sharing)
- Run the following command

`python3 start.py --test --image-path="out" --model-path=out/models/sample_trained --test-content-path="out/sample.json" --no-attention --log`


To train a new model from scratch:
1. Prepare the proper data annotation and split using `../DataAnalysis/split_data.py`
2. Adjust the config parameters on `start.py`, including vocab version.
3. Use the following command

`python3 start.py --image-path=<path to images> --test-content-path="<path to test.json>" --train-content-path="<path to train.json>" "`
