# Load packages
import tensorflow.keras as tfk
from kerastuner import HyperModel

# Hyperparameter tuning
class classifierHyperModel(HyperModel):
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build(self, hp):
        model = tfk.Sequential()
        model.add(
            tfk.layers.BatchNormalization(
                input_shape=(self.input_shape,)
            ))
        model.add(
            tfk.layers.Dense(
            units=hp.Int(
                'units_1',
                #placeholder values for now
                min_value=32,
                max_value=512,
                step=32,
                default=128
            ),
            activation=hp.Choice(
                'dense_activation_1',
                values=['elu','relu','tanh','sigmoid'],
                default='elu'
            ))
        )
        model.add(
            tfk.layers.Dense(
            units=hp.Int(
                'units_2',
                #placeholder values for now
                min_value=32,
                max_value=512,
                step=32,
                default=128
            ),
            activation=hp.Choice(
                'dense_activation_2',
                values=['elu','relu','tanh','sigmoid'],
                default='elu'
            ))
        )
        model.add(
            tfk.layers.Dense(
            units=hp.Int(
                'units_3',
                #placeholder values for now
                min_value=32,
                max_value=512,
                step=32,
                default=128
            ),
            activation=hp.Choice(
                'dense_activation_3',
                values=['elu','relu','tanh','sigmoid'],
                default='elu'
            ))
        )
        model.add(
            tfk.layers.Dropout(rate=hp.Float(
                'dropout',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05
            ))
        )
        model.add(
            tfk.layers.Dense(
            units=hp.Int(
                'units_4',
                #placeholder values for now
                min_value=32,
                max_value=512,
                step=32,
                default=128
            ),
            activation=hp.Choice(
                'dense_activation_4',
                values=['elu','relu','tanh','sigmoid'],
                default='elu'
            ))
        )
        model.add(
            tfk.layers.Dense(
            units=hp.Int(
                'units_5',
                #placeholder values for now
                min_value=32,
                max_value=512,
                step=32,
                default=128
            ),
            activation=hp.Choice(
                'dense_activation_5',
                values=['elu','relu','tanh','sigmoid'],
                default='elu'
            ))
        )
        model.add(
            tfk.layers.Dense(
            units=hp.Int(
                'units_6',
                #placeholder values for now
                min_value=32,
                max_value=512,
                step=32,
                default=128
            ),
            activation=hp.Choice(
                'dense_activation_6',
                values=['elu','relu','tanh','sigmoid'],
                default='elu'
            ))
        )
        model.add(
            tfk.layers.Dense(
                self.num_classes,activation="softmax"
            ))
        
        model.compile(
            optimizer=tfk.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=5e-4
                )
            ),
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )
        return model
