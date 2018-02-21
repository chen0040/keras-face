from keras import backend as K

K.set_image_data_format('channels_first')
from keras_face.library.fr_utils import *
from keras_face.library.inception_blocks_v2 import *

np.set_printoptions(threshold=np.nan)


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = K.sum(K.maximum(basic_loss, 0))

    return loss


def triplet_loss_test():
    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
        loss = triplet_loss(y_true, y_pred)

        print("loss = " + str(loss.eval()))


class FaceNet(object):
    def __init__(self):
        self.model = None

    def load_model(self, model_dir_path):
        self.model = faceRecoModel(input_shape=(3, 96, 96))
        print("Total Params:", self.model.count_params())
        self.model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(self.model, model_dir_path)

    def img_to_encoding(self, image_path):
        return img_to_encoding(image_path, self.model)

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

        if threshold is None:
            threshold = 0.7

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = img_to_encoding(image_path, self.model)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        dist = float(np.linalg.norm(encoding - database[identity]))

        # Step 3: Open the door if dist < threshold, else don't open (≈ 3 lines)
        if dist < threshold:
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

        if threshold is None:
            threshold = 0.7

        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        encoding = img_to_encoding(image_path, self.model)

        ## Step 2: Find the closest encoding ##

        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        identity = None

        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():

            # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
            dist = np.linalg.norm(db_enc - encoding)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > threshold:
            print("Not in the database.")
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity


def main():
    triplet_loss_test()

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

    fnet.verify(image_dir_path + "/camera_0.jpg", "younes", database)
    fnet.verify(image_dir_path + "/camera_2.jpg", "kian", database)
    fnet.who_is_it(image_dir_path + "/camera_0.jpg", database)


if __name__ == '__main__':
    main()
