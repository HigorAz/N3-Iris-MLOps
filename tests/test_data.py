from src import data


def test_carrega_dados():
    cfg = data.carrega_config()
    X, y = data.carrega_dados(cfg)
    assert list(X.columns) == data.FEATURES
    assert len(X) == len(y) == 150
    assert y.nunique() == 3


def test_prepara_split():
    cfg = data.carrega_config()
    X_tr, X_te, y_tr, y_te, scaler, encoder = data.prepara(cfg)
    assert len(X_tr) + len(X_te) == 150
    assert X_tr.shape[1] == 4
    assert len(encoder.classes_) == 3
