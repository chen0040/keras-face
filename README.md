# keras-face

face verification and recognition using Keras

The project contains two implementations: [DeepFace](keras_face/library/face_net.py) and 
[VGG16 + Siamese](keras_face/library/siamese.py)

* DeepFace: [keras_face/library/face_net.py](keras_face/library/face_net.py) contains the deep face implementation tought in the
coursea course deeplearning.ai
* VGG16 + Siamese: [keras_face/library/siamese.py](keras_face/library/siamese.py) is my own implementation of siamese network + VGG16 + 
contrastive loss function (image similarity function)

# Usage

### DeepFace 

Below shows the [sample codes](demo/face_net_demo.py) which verifies whether a particular camera image is a person in an image database or
whether a particular camera image is which person in the image database (or not at all)

```python
from keras_face.library.face_net import FaceNet


def main():
    model_dir_path = './models'
    image_dir_path = "./data/images"

    fnet = FaceNet()
    fnet.load_model(model_dir_path)

    database = {}
    database["danielle"] = fnet.img_to_encoding(image_dir_path + "/danielle.png")
    database["younes"] = fnet.img_to_encoding(image_dir_path + "/younes.jpg")
    database["tian"] = fnet.img_to_encoding(image_dir_path + "/tian.jpg")
    database["andrew"] = fnet.img_to_encoding(image_dir_path + "/andrew.jpg")
    database["kian"] = fnet.img_to_encoding(image_dir_path + "/kian.jpg")
    database["dan"] = fnet.img_to_encoding(image_dir_path + "/dan.jpg")
    database["sebastiano"] = fnet.img_to_encoding(image_dir_path + "/sebastiano.jpg")
    database["bertrand"] = fnet.img_to_encoding(image_dir_path + "/bertrand.jpg")
    database["kevin"] = fnet.img_to_encoding(image_dir_path + "/kevin.jpg")
    database["felix"] = fnet.img_to_encoding(image_dir_path + "/felix.jpg")
    database["benoit"] = fnet.img_to_encoding(image_dir_path + "/benoit.jpg")
    database["arnaud"] = fnet.img_to_encoding(image_dir_path + "/arnaud.jpg")

    # verifies whether a particular camera image is a person in the image database
    dist, is_valid = fnet.verify(image_dir_path + "/camera_0.jpg", "younes", database)
    print('camera_0.jpg is' + (' ' if is_valid else ' not ') + 'yournes')
    dist, is_valid = fnet.verify(image_dir_path + "/camera_2.jpg", "kian", database)
    print('camera_0.jpg is' + (' ' if is_valid else ' not ') + 'yournes')
    
    # whether a particular camera image is which person in the image database (or not at all)
    dist, identity = fnet.who_is_it(image_dir_path + "/camera_0.jpg", database)
    if identity is None:
        print('camera_0.jpg is not found in database')
    else:
        print('camera_0.jpg is ' + str(identity))


if __name__ == '__main__':
    main()
```

### VGG16 + Siamese

Below shows [sample codes](demo/siamese_demo_train.py) how to train the V166+Siamese network:

```python
from keras_face.library.siamese import SiameseFaceNet


def main():
    fnet = SiameseFaceNet()
    fnet.vgg16_include_top = True # default is False

    model_dir_path = './models'
    image_dir_path = "./data/images"

    database = dict()
    database["danielle"] = [fnet.img_to_encoding(image_dir_path + "/danielle.png")]
    database["younes"] = [fnet.img_to_encoding(image_dir_path + "/younes.jpg")]
    database["tian"] = [fnet.img_to_encoding(image_dir_path + "/tian.jpg")]
    database["andrew"] = [fnet.img_to_encoding(image_dir_path + "/andrew.jpg")]
    database["kian"] = [fnet.img_to_encoding(image_dir_path + "/kian.jpg")]
    database["dan"] = [fnet.img_to_encoding(image_dir_path + "/dan.jpg")]
    database["sebastiano"] = [fnet.img_to_encoding(image_dir_path + "/sebastiano.jpg")]
    database["bertrand"] = [fnet.img_to_encoding(image_dir_path + "/bertrand.jpg")]
    database["kevin"] = [fnet.img_to_encoding(image_dir_path + "/kevin.jpg")]
    database["felix"] = [fnet.img_to_encoding(image_dir_path + "/felix.jpg")]
    database["benoit"] = [fnet.img_to_encoding(image_dir_path + "/benoit.jpg")]
    database["arnaud"] = [fnet.img_to_encoding(image_dir_path + "/arnaud.jpg")]

    fnet.fit(database=database, model_dir_path=model_dir_path)

if __name__ == '__main__':
    main()

```

Below shows the [sample codes](demo/siamese_demo_train.py) which verifies whether a particular camera image is a person in an image database or
whether a particular camera image is which person in the image database (or not at all)

```python
from keras_face.library.siamese import SiameseFaceNet


def main():
    fnet = SiameseFaceNet()

    model_dir_path = './models'
    image_dir_path = "./data/images"
    fnet.load_model(model_dir_path)

    database = dict()
    database["danielle"] = [fnet.img_to_encoding(image_dir_path + "/danielle.png")]
    database["younes"] = [fnet.img_to_encoding(image_dir_path + "/younes.jpg")]
    database["tian"] = [fnet.img_to_encoding(image_dir_path + "/tian.jpg")]
    database["andrew"] = [fnet.img_to_encoding(image_dir_path + "/andrew.jpg")]
    database["kian"] = [fnet.img_to_encoding(image_dir_path + "/kian.jpg")]
    database["dan"] = [fnet.img_to_encoding(image_dir_path + "/dan.jpg")]
    database["sebastiano"] = [fnet.img_to_encoding(image_dir_path + "/sebastiano.jpg")]
    database["bertrand"] = [fnet.img_to_encoding(image_dir_path + "/bertrand.jpg")]
    database["kevin"] = [fnet.img_to_encoding(image_dir_path + "/kevin.jpg")]
    database["felix"] = [fnet.img_to_encoding(image_dir_path + "/felix.jpg")]
    database["benoit"] = [fnet.img_to_encoding(image_dir_path + "/benoit.jpg")]
    database["arnaud"] = [fnet.img_to_encoding(image_dir_path + "/arnaud.jpg")]

    fnet.verify(image_dir_path + "/camera_0.jpg", "younes", database)
    fnet.verify(image_dir_path + "/camera_2.jpg", "kian", database)
    fnet.who_is_it(image_dir_path + "/camera_0.jpg", database)


if __name__ == '__main__':
    main()
```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 

# Todo


For VGG16 + Siamese, the training was not well-done as there are currently very limited number of sample images used for
training (only 12 images for 12 persons). Ideally, need to train using 100,000 images for 10,000 persons. Will need
to add in larger dataset for the training

# Note
For DeepFace (namely [keras_face/library/face_net.py](keras_face/library/face_net.py)), some utility classes 
and weights are taken from [https://github.com/shahariarrabby/deeplearning.ai](https://github.com/shahariarrabby/deeplearning.ai)
, also it contains only the prediction part
