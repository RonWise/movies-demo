import joblib


class Classifier(object):
    def __init__(self):
        self.mlb = joblib.load("movies_mlb.pkl")
        self.model = joblib.load("movies_model.pkl")
        self.class_name = [str(genre) for genre in self.mlb.classes_]

    def argmax(self, data: list):
        f = lambda i: data[i]
        return max(range(len(data)), key=f)

    def predict_text(self, text):
        try:
            prediction_prob = self.model.predict_proba([text])
            pred_prob = []
            for pred_genre in prediction_prob:
                pred_prob.append(pred_genre[:, 1].tolist())

            result = self.class_name[self.argmax(pred_prob)]
            return result
        except:
            print("prediction error")
            return None

    def get_result_message(self, text):
        prediction = self.predict_text(text)
        return prediction
