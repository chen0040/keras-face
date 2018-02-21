import random

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
import os


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class SiameseFaceNet(object):
    model_name = 'siamese-face-net'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.vgg16_include_top = False

        self.labels = None
        self.config = None
        self.input_shape = None
        self.threshold = 0.5
        self.vgg16_model = None

    def img_to_encoding(self, image_path):
        print('encoding: ', image_path)
        if self.vgg16_model is None:
            self.vgg16_model = self.create_vgg16_model()

        image = cv2.imread(image_path, 1)
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        return self.vgg16_model.predict(input)

    def load_model(self, model_dir_path):
        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)
        self.config = np.load(config_file_path).item()
        self.labels = self.config['labels']
        self.input_shape = self.config['input_shape']
        self.threshold = self.config['threshold']
        self.vgg16_include_top = self.config['vgg16_include_top']

        self.vgg16_model = self.create_vgg16_model()
        self.model = self.create_network(input_shape=self.input_shape)
        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_base_network(self, input_shape):
        '''Base network to be shared (eq. to feature extraction).
        '''
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < self.threshold, y_true.dtype)))

    def create_network(self, input_shape):
        # network definition
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=[self.accuracy])

        print(model.summary())

        return model

    def create_pairs(self, database, names):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        num_classes = len(database)
        pairs = []
        labels = []
        n = min([len(database[name]) for name in database.keys()])
        for d in range(len(names)):
            name = names[d]
            x = database[name]
            for i in range(n):
                pairs += [[x[i], x[(i + 1) % n]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = x[i], database[names[dn]][i]
                pairs += [[z1, z2]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-architecture.h5'

    def create_vgg16_model(self):
        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        return vgg16_model

    def fit(self, database, model_dir_path, epochs=None, batch_size=None, threshold=None, vgg16_include_top=None):
        if threshold is not None:
            self.threshold = threshold
        if batch_size is None:
            batch_size = 128
        if epochs is None:
            epochs = 20
        if vgg16_include_top is not None:
            self.vgg16_include_top = vgg16_include_top

        for name, feature in database.items():
            self.input_shape = feature[0].shape
            break

        self.vgg16_model = self.create_vgg16_model()

        self.model = self.create_network(input_shape=self.input_shape)
        architecture_file_path = self.get_architecture_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        names = []
        self.labels = dict()
        for name in database.keys():
            names.append(name)
            self.labels[name] = len(self.labels)

        self.config = dict()
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold
        self.config['vgg16_include_top'] = self.vgg16_include_top

        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)

        t_x, t_y = self.create_pairs(database, names)

        print('data set pairs: ', t_x.shape)

        self.model.fit([t_x[:, 0], t_x[:, 1]], t_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2,
                       verbose=SiameseFaceNet.VERBOSE,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

    def verify(self, image_path, identity, database, threshold=None):
        """
        Function that verifies if the person on the "image_path" image is "identity".

        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """

        if threshold is not None:
            self.threshold = threshold

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = self.img_to_encoding(image_path)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        input_pairs = []
        x = database[identity]
        for i in range(len(x)):
            input_pairs.append([encoding, x[i]])
        input_pairs = np.array(input_pairs)
        dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]

        # Step 3: Open the door if dist < threshold, else don't open (≈ 3 lines)
        if dist < self.threshold:
            print("It's " + str(identity))
            is_valid = True
        else:
            print("It's not " + str(identity))
            is_valid = False

        return dist, is_valid

    def who_is_it(self, image_path, database, threshold=None):
        """
        Implements face recognition for the happy house by finding who is the person on the image_path image.

        Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """

        if threshold is not None:
            self.threshold = threshold

        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        encoding = self.img_to_encoding(image_path)

        ## Step 2: Find the closest encoding ##

        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        identity = None

        # Loop over the database dictionary's names and encodings.
        for (name, x) in database.items():

            input_pairs = []
            for i in range(len(x)):
                input_pairs.append([encoding, x[i]])
            input_pairs = np.array(input_pairs)
            dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]

            print("--for " + str(name) + ", the distance is " + str(dist))

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > self.threshold:
            print("Not in the database.")
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity


def main():
    fnet = SiameseFaceNet()
    fnet.vgg16_include_top = True

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

    fnet.load_model(model_dir_path)
    fnet.verify(image_dir_path + "/camera_0.jpg", "younes", database)
    fnet.verify(image_dir_path + "/camera_2.jpg", "kian", database)
    fnet.who_is_it(image_dir_path + "/camera_0.jpg", database)


if __name__ == '__main__':
    main()
