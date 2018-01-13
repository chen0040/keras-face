# keras-face

face verification and recognition using Keras

The project contains two implementations which can be founds in 
[keras_face/library/face_net.py](keras_face/library/face_net.py) and 
[keras_face/library/siamese.py](keras_face/library/siamese.py)

[keras_face/library/face_net.py](keras_face/library/face_net.py) contains the deep face implementation tought in the
coursea course deeplearning.ai

[keras_face/library/siamese.py](keras_face/library/siamese.py) is my own implementation of siamese network + VGG16 + 
binary crossentropy loss function (image similarity function)

# Usage

Below shows the sample codes which verifies whether a particular camera image is a person in an image database or
whether a particular camera image is which person in the image database (or not at all)

```python
from keras_face.library.face_net import FaceNet


def main():
    model_dir_path = '../training/models'
    image_dir_path = "../training/data/images"

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

# Note

some utility classes and weights are taken from [https://github.com/shahariarrabby/deeplearning.ai](https://github.com/shahariarrabby/deeplearning.ai)
