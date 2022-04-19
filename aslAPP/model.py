import tensorflow as tf
import numpy as np

class Model:

    def ___init__(self):
        self.model_interpreter = tf.lite.Interpreter(model_path="static\tflite_model\tflite_classifier.tflite")
        self.model_interpreter.allocate_tensors()
        self.input_details = self.model_interpreter.get_input_details()
        self.output_details = self.model_interpreter.get_output_details()
        self.labels = {"1": "a",
        "2": "b",
        "3": "x",
        "4": "d",
        "5": "e",
        "6": "f",
        "7": "g",
        "8": "h",
        "9": "i",
        "10": "j",
        "11": "k",
        "12": "l",
        "13": "m",
        "14": "n",
        "15": "o",
        "16": "p",
        "17": "q",
        "18": "r",
        "19": "s",
        "20": "t",
        "21": "u",
        "22": "v",
        "23": "w",
        "24": "x",
        "25": "y",
        "26": "z"}
        
    def __call__(self, landmarks):
        try:
            self.model_interpreter.set_tensor(self.input_details[0]["index"], np.reshape(np.array(landmarks), dtype=np.float32), (1, 42))
            self.model_interpreter.invoke()
            prediction = self.model_interpreter.get_tensor(self.output_details[0]["index"])
            return np.argmax(np.squeeze([prediction]))
        except:
            raise Exception("Not")
    
    def predict(self, landmarks):
        self.model_interpreter.set_tensor(self.input_details[0]["index"], np.reshape(np.array(landmarks), dtype=np.float32), (1, 42))
        self.model_interpreter.invoke()
        prediction = self.model_interpreter.get_tensor(self.output_details[0]["index"])
        return np.argmax(np.squeeze([prediction]))