# Data Extraction and Exploration

Three main functionalities of this module are:
1. Icon extraction
2. Icon filteration
3. Empirical study
4. Split data for training 


1. To extracts icons from a raw dataset of xml layouts:
`python3 -u icon_extractor.py --data-path="<root directory of LabelDroid dataset>" --step=<0:3> --version=1`

2. To filter candidate icons:
`python3 -u filter_icons.py --icon-version=1 --step=<0:3> --filter-version=1 --data-path="<root directory of LabelDroid dataset>"`

3. To get the results of studying icons dataset:
    1. label distribution:
`python3 -u empirical_study.py --label-dist --filter-version=1 --image-path=<root directory of icon images>`
    2. image distribution:
    `python3 empirical_study.py --image-dist --filter-version=1 --image-path=<root directory of icon images>`
    3. context distribution:
    `python3 empirical_study.py --context-dist --filter-version=1 --image-path=<root directory of icon images>`

4. To split data to test/train/val and write in proper annotation:
`python3 -u split_data.py --filter-version=1 --split-version=1 --ignore-dup --data-path="<root directory of LabelDroid dataset>"`