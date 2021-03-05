
# DCASE 2020: Sound event localization and detection (SELD) task


## DATASETS

The participants can choose either of the two or both the following datasets,

 * **TAU-NIGENS Spatial Sound Events 2020 - Ambisonic**
 * **TAU-NIGENS Spatial Sound Events 2020 - Microphone Array**

These datasets contain recordings from an identical scene, with **TAU-NIGENS Spatial Sound Events 2020 - Ambisonic** providing four-channel First-Order Ambisonic (FOA) recordings while  **TAU-NIGENS Spatial Sound Events 2020 - Microphone Array** provides four-channel directional microphone recordings from a tetrahedral array configuration. Both formats are extracted from the same microphone array, and additional information on the spatial characteristics of each format can be found below. The participants can choose one of the two, or both the datasets based on the audio format they prefer. Both the datasets, consists of a development and evaluation set. 

The development set consists of 600, one minute long recordings sampled at 24000 Hz. All participants are expected to use the fixed splits provided in the baseline method for reporting the development scores. We use 400 recordings for training split (fold 3 to 6), 100 for validation (fold 2) and 100 for testing (fold 1). The evaluation set consists of 200, one-minute recordings, and is released during the evaluation phase of the challenge. After the end of the DCASE2020 Challenge, the unknown labels of the evaluation set are also released for further testing and comparison of methods out of the challenge, with the challenge results.

The development and evaluation datasets in both formats can be downloaded from the link

[**Download the dataset**](https://doi.org/10.5281/zenodo.4064792)


### Training the SELDnet

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset name and its path in `parameter.py` script. For the above example, you will change `dataset='foa'` and `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped. 

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. Run the script as shown below. This will dump the normalized features and labels in the `feat_label_dir` folder.

```
python3 batch_feature_extraction.py
```

You can now train the SELDnet using default parameters using
```
python3 seld_torch.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following
```
python3 seld_torch.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

In order to get baseline results on the development set for Microphone array recordings, you can run the following command
```
python3 seld_torch.py 2
```
Similarly, for Ambisonic format baseline results, run the following command
```
python3 seld_torch.py 4
```

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on only 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `parameter.py` script and train on the entire data.

* The code also plots training curves, intermediate results and saves models in the `model_dir` path provided by the user in `parameter.py` file.

* In order to visualize the output of SELDnet and for submission of results, set `dcase_output=True` and provide `dcase_dir` directory. This will dump file-wise results in the directory, which can be individually visualized using `visualize_SELD_output.py` script.
