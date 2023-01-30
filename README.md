# SPOTER Embeddings

This repository provides code for the SPOTER embedding model.
<!-- explained in this [blog post](link...). -->
The model is built the Sign Pose-Based Transformer [SPOTER] approach presented in
[Sign Pose-Based Transformer for Word-Level Sign Language Recognition](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html)but with a slight modification - 
it operates as an embedding model instead of a classification model.
This enables zero-shot capabilities on new Sign Language datasets from around the world.
<!-- More details about this are shown in the blog post mentioned above. -->

<!-- ## Results -->
<!-- Include some graphical results here -->

<!--  Also link the product blog here -->

## Getting Started

The best way of running code from this repo is by using **Docker**.

Clone this repository and run:
```
docker build -t spoter_embeddings .
docker run --rm -it --entrypoint=bash --gpus=all -v $PWD:/app spoter_embeddings
```

> Running without specifying the `entrypoint` will train the model with the hyperparameters specified in `train.sh`

If you prefer running in a **virtual environment**, then start by installing dependencies:

```shell
pip install -r requirements.txt
```

To train the model, run `train.sh` in Docker or your virtual env.

The hyperparameters with their descriptions can be found in the [train.py](link...) file.

## Data

Same as with SPOTER, this model works on top of sequences of signers' skeletal data extracted from videos.
This means that the input data has a much lower dimension compared to using videos directly, and therefore the model is
quicker and lighter, while you can choose any SOTA body pose model to preprocess video.
This results in a much lower input dimension, making the model faster and more efficient, with real-time processing capabilities (e.g. processinga 4-second 25 FPS video takes approximately 40ms using onnxruntime within a web browser).

![Alt Text](http://spoter.signlanguagerecognition.com/img/datasets_overview.gif)

For ready-to-use datasets refer to the [SPOTER] repository.

For optimum results, we recommend building your own dataset by downloading a Sign language video dataset such as [WLASL] and then using the `extract_mediapipe_landmarks.py` and `create_wlasl_landmarks_dataset.py` scripts to create body keypoints datasets that can be used to train the Spoter embeddings model.

You can run these scripts as follows:
```bash
# This will extract landmarks from the downloaded videos
python3 preprocessing.py extract -videos <path_to_video_folder> --output-landmarks <path_to_landmarks_folder>

# This will create a dataset (csv file) with the first 100 classes, splitting 20% of it to the test set, and 80% for train
python3 preprocessing.py create -videos <path_to_video_folder> -lmks <path_to_landmarks_folder> --dataset-folder=<output_folder> --create-new-split -ts=0.2
```

## Example notebooks
There are two Jupyter notebooks included in the `notebooks` folder.
* embeddings_evaluation.ipynb: This notebook shows how to evaluate a model
* visualize_embeddings.ipynb: Model embeddings visualization, optionally with embedded input video


## Modifications on [SPOTER](https://github.com/matyasbohacek/spoter)
Here is a list of the main modifications made on SPOTER code and model architecture:

* The output layer is a linear layer but trained using triplet loss instead of CrossEntropyLoss. The output of the model
is therefore an embedding vector that can be used for several downstream tasks.
* We started using the keypoints dataset published by SPOTER but later created new datasets using BlazePose from Mediapipe (as seen in [Spoter 2](https://arxiv.org/abs/2210.00893)). This improves results considerably.
* We select batches so that they contain several hard triplets and then compute the loss on all hard triplets found in each batch.
* We implemented some code refactoring to acomodate new classes.
* Minor code fix when using rotate augmentation to avoid exceptions.

## Tracking experiments with ClearML
The code supports tracking experiments, datasets, and models in a ClearML server.
If you want to do this make sure to pass the following arguments to train.py:

```
    --dataset_loader=clearml
    --tracker=clearml
```

Also make sure to correctly configure your clearml.conf file.
If using Docker, you can map it into Docker adding these volumes when running `docker run`:

```
-v $HOME/clearml.conf:/root/clearml.conf -v $HOME/.clearml:/root/.clearml
```


## License

The **code** is published under the [Apache License 2.0](./LICENSE) which allows for both academic and commercial use if
relevant License and copyright notice is included, our work is cited and all changes are stated.

The license for the [WLASL](https://arxiv.org/pdf/1910.11006.pdf) and [LSA64](https://core.ac.uk/download/pdf/76495887.pdf) datasets used for experiments is, however, the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license which allows only for non-commercial usage.


[SPOTER]: (https://github.com/matyasbohacek/spoter)
[WLASL]: (https://dxli94.github.io/WLASL/)
