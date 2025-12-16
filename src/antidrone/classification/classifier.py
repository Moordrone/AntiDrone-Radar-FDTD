class DummyClassifier:
    def predict(self, X):
        return ['drone' for _ in X]
