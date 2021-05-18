import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import logging.handlers
from typing import List
from pathlib import Path


logger = logging.getLogger('ml')
def setup_logging():
    screen_formatter = logging.Formatter('[%(levelname)s] - %(message)s')
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(screen_formatter)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler('ml.log', maxBytes=100000, backupCount=10)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(screen_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

class BertClassifierModel():
    def __init__(self, random_state=44, 
                        autotune = tf.data.AUTOTUNE,
                        batch_size = 32, test_size=0.2):
        setup_logging()
        self.RANDOM_STATE = random_state
        self.AUTOTUNE = autotune
        self.BATCH_SIZE = batch_size
        self.TEST_SIZE = test_size

        self.model=None
        self.dataset_train=None
        self.dataset_val=None
        self.history=None

        self.threshold=0.5 #prediction threshold

    def load_df(self, filepath, index_col, text_col, target_col):
        """
        1. load pandas dataframe with text and label column 
        2. seperate loaded data into train and validation set
        3. convert to tensorflow dataset and arrange for training(shuffle/batch/cache/prefetch)

        filepath: filepath loaded into pandas dataframe
        index_col: index column name of the dataframe
        text_col: column name of text data
        target_col: label of data
        """
        try:
            logger.info(f'loading csv {filepath}, index col = {index_col}')
            df = pd.read_csv(filepath, index_col=index_col)
            text_data = df.pop(text_col)
            target_data = df.pop(target_col)
            
            logger.info(f'splitting train/test. test size = {self.TEST_SIZE}')
            X_train, X_val, y_train, y_val = train_test_split(
                text_data, target_data, test_size=self.TEST_SIZE,
                random_state=self.RANDOM_STATE, 
            )

            raw_dataset_train = tf.data.Dataset.from_tensor_slices(
                (X_train, y_train)
            )

            raw_dataset_val = tf.data.Dataset.from_tensor_slices(
                (X_val, y_val)
            )

            self.dataset_train = raw_dataset_train.shuffle(
                len(raw_dataset_train)
            ).batch(
                batch_size=self.BATCH_SIZE
            ).cache().prefetch(buffer_size=self.AUTOTUNE)
    
            self.dataset_val = raw_dataset_train.shuffle(
                len(raw_dataset_val)
            ).batch(
                batch_size=self.BATCH_SIZE
            ).cache().prefetch(buffer_size=self.AUTOTUNE)
                

        except:
            logger.error('Unexpected error while loading data:', exc_info=True)


    def build_model(self,
                    preprocess_handle='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                    encoder_handle='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
                    drop_out=0.1):
        """
        create bert model from TensorFlow Hub
        https://www.tensorflow.org/tutorials/text/classify_text_with_bert#export_for_inference

        preprocess_handle: matching preprocessor with bert encoder. no additional preprocessing is needed 
                            according to the document above
        encoder_handle: bert model 
        """
        try:
            logger.info(f'Building model. preprocessor={preprocess_handle}, encoder={encoder_handle}')
            text_input = tf.keras.layers.Input(shape=(),
            dtype=tf.string, name='text')
            preprocess_layer = hub.KerasLayer(preprocess_handle, name='preprocessing')
            encoder_inputs = preprocess_layer(text_input)
            encoder = hub.KerasLayer(encoder_handle, trainable=True, name='bert_encoder')
            outputs = encoder(encoder_inputs)
            net = outputs['pooled_output']
            net = tf.keras.layers.Dropout(drop_out)(net)
            net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

            self.model = tf.keras.Model(text_input, net)
            logger.info(f'Model build succeeded.')
        except:
            logger.error('Unexpected error while building model:', exc_info=True)

    def train(self, num_epochs = 5, lr=3e-5, optimizer_name='adamw'):
        """
        train the model 
        num_epochs: number of epochs
        lr: learning rate
        optimizer_name: optimizer 

        returns History object from model.fit()
        """
        try:
            logger.info(f'training started... num_epochs={num_epochs}, lr={lr}, optimizer={optimizer_name}')
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = tf.metrics.BinaryAccuracy()
            steps_per_epoch = tf.data.experimental.cardinality(
                self.dataset_train
            ).numpy()
            num_train_steps = steps_per_epoch * (num_epochs)
            num_warmup_steps = int(0.1*num_train_steps)
            
            optimizer = optimization.create_optimizer(
                init_lr = lr,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                optimizer_type=optimizer_name
            )

            self.model.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics)
            self.history = self.model.fit(
                x=self.dataset_train,
                validation_data=self.dataset_val,
                epochs=num_epochs
            )
            logger.info('training completed.')
            return self.history
        except:
            logger.error('Unexpected error while training:', exc_info=True)

    def save(self, model_path='./saved'):
        """
        saved the (trained) model for future use. ./saved as a default location
        model_path: location of the save
        """
        try:
            self.model.save(model_path, include_optimizer=False)
        except:
            logger.error('Unexpected error while saving:', exc_info=True)
    
    def reload(self, model_path):
        """
        reload pre-trained model 
        model_path: location of the saved model
        """
        try:
            logger.info(f'Loading pre-trained model from {model_path}')
            self.model = tf.saved_model.load(model_path)
            logger.info(f'load successful')
        except:
            logger.error('Unexpected error while reloading:', exc_info=True)
        
    def predict(self, inputs:List[str]):
        """
        make a prediction and return predicted labels
        inputs: list of strings to label
        
        returns: list of labels[0,1]
        """
        try:
            logger.info(f'predictions on the input')
            predicted = tf.sigmoid(self.model(tf.constant(inputs)))
            return list(map(lambda x: 0 if x[0] < self.threshold else 1, predicted.numpy()))

        except:
            logger.error('Unexpected error while predicting:', exc_info=True)

    def evaluate(self, test_ds):
        """
        print model evaluation for internal use
        test_ds: pre labeled test data for evaluation
        """
        loss, accuracy = self.model.evaluate(test_ds)

        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')


model = BertClassifierModel()
save_path = Path(__file__).parent / "saved"
model.reload(str(save_path))


def get_model():
    return model