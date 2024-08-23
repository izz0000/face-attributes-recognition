# How To Use

## Training a Model
1. Fill `KAGGLE_USERNAME` & `KAGGLE_KEY` environment variables in `.env` file
2. Run in your terminal: `python src/train.py [your training arguments as following]`

### Training Parameters
|     Parameter      |                       Description                       |                                                    Data Type / Choices                                                     |       Default        |
|:------------------:|:-------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|:--------------------:|
| output-layer-depth |           number of the FC layers before end            |                                                            int                                                             |          4           |
|      backbone      |          the pretrained backbone of the model           |                      `mobilenet-v3`, `resnet50`, `inception3`, `efficientnet`, `densenet121` `vgg16`                       |    `mobilenet-v3`    |
|      data-dir      |                   path of data folder                   |                                                           string                                                           |      `../data/`      |
|     batch-size     |                  the known batch size                   |                                                            int                                                             |          32          |
|    accelerator     |                   computation device                    |                                                   `cpu`, `gpu`, or `tpu`                                                   |  `gpu` if available  |
|     precision      |                  float precision mode                   | `transformer-engine`, `transformer-engine-float16`, `16-true`, `16-mixed`, `bf16-true`, `bf16-mixed`, `32-true`, `64-true` | depends on the model |
|       epochs       |                    the known epochs                     |                                                            int                                                             |          5           |
|   learning-rate    |                 the known learning rate                 |                                                           float                                                            |        0.001         |
|    num-workers     | the number of processes that loads the data in parallel |                                                            int                                                             |          2           |
|     optimizer      |                   the known optimizer                   |                                                  `adam`, `sgd`, `rmsprop`                                                  |        `adam`        |



## Predict from Image
1. Get trained model id
2. Run in your terminal: `python src/predict.py <your_model_id> <your_image_path>`