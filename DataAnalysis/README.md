# Data Extraction and Exploration

Three main functionalities of this module are:
1. Icon extraction
2. Icon filteration
3. Empirical study
4. Split data for training 

====
For the raw data, you can either download the whole raw dataset of [LabelDroid](https://www.dropbox.com/sh/kfkhevxykzwputb/AAAhL6ipmOg4zZn4jUL_myF0a?dl=0), or the small [sample set](https://drive.google.com/file/d/13jrdZoJPLZivTsl_hOd9vgHV7Jvt04Ol/view?usp=sharing) of that which was created to evaluate the functionality of the source code.

1. To extracts icons from a raw dataset of xml layouts:
`python3 icon_extractor.py --data-path="<root directory of LabelDroid dataset>" --step=<0:3> --version=1`

2. To filter candidate icons:
`python3 filter_icons.py --icon-version=1 --step=<0:3> --filter-version=1 --data-path="<root directory of LabelDroid dataset>"`

3. To get the results of studying icons dataset:
    
    e.g. label distribution:
`python3 empirical_study.py --label-dist --filter-version=1 --image-path=<root directory of icon images>`
    
4. To split data to test/train/val and write in proper annotation:
`python3 split_data.py --filter-version=1 --split-version=1 --ignore-dup --data-path="<root directory of LabelDroid dataset>"`
