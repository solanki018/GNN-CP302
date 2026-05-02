import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mendeleev import element
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
import re

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ================= PROPERTY CACHE =================
PROPERTY_CACHE = {}

D_BAND_FILLING = {"Ag":1.0,"Au":1.0,"Cd":1.0,"Co":0.8,"Cu":1.0,"Fe":0.7,"Ir":0.8,
"Ni":0.9,"Os":0.7,"Pd":0.9,"Pt":0.9,"Rh":0.8,"Ru":0.7,"Zn":1.0}

ENTHALPY_FUSION = {"Ag":105,"Au":64,"Cd":112,"Co":272,"Cu":205,"Fe":247,"Ir":266,
"Ni":290,"Os":134,"Pd":157,"Pt":113,"Rh":258,"Ru":381,"Zn":112}

WORK_FUNCTION = {"Ag":4.26,"Au":5.10,"Cd":4.08,"Co":5.00,"Cu":4.65,"Fe":4.67,"Ir":5.67,
"Ni":5.15,"Os":5.93,"Pd":5.12,"Pt":5.65,"Rh":4.98,"Ru":4.71,"Zn":4.33}

COHESIVE_ENERGY = {"Ag":2.95,"Au":3.81,"Cd":1.16,"Co":4.39,"Cu":3.49,"Fe":4.28,"Ir":6.94,
"Ni":4.44,"Os":8.17,"Pd":3.89,"Pt":5.84,"Rh":5.75,"Ru":6.74,"Zn":1.35}

def get_mendeleev_data(symbol):
    if symbol in PROPERTY_CACHE:
        return PROPERTY_CACHE[symbol]
    try:
        el = element(symbol)
        data = {
            "en": float(el.en_pauling or 2.5)/4,
            "group": (el.group_id or 0)/18,
            "ie": (el.ionization_energies[0] if el.ionization_energies else 7)/15,
            "ea": (el.electron_affinity or 0.5)/4,
            "mp": (el.melting_point or 1000)/4000,
            "bp": (el.boiling_point or 2000)/6000,
            "d_band": D_BAND_FILLING.get(symbol,0.5),
            "hfus": ENTHALPY_FUSION.get(symbol,150)/400,
            "wf": WORK_FUNCTION.get(symbol,4.5)/7,
            "cohesive": COHESIVE_ENERGY.get(symbol,3)/9
        }
    except:
        data = {k:0.5 for k in ["en","group","ie","ea","mp","bp","d_band","hfus","wf","cohesive"]}
    PROPERTY_CACHE[symbol] = data
    return data

# ================= SURFACE FEATURE =================
def get_surface_vector(metal, facet):
    p = get_mendeleev_data(metal)

    m_feat = [
        p["group"], p["en"], p["ie"], p["ea"],
        p["mp"], p["bp"], p["d_band"],
        p["hfus"], p["wf"], p["cohesive"]
    ]

    facet_digits = "".join(re.findall(r'\d', str(facet)))
    f_str = facet_digits.ljust(4,'0')
    f_feat = [float(d)/9.0 for d in f_str[:4]]

    return np.array(m_feat + f_feat, dtype=np.float32)

# ================= DATA =================
def create_dataset(df):
    X = np.array([
        get_surface_vector(m,f)
        for m,f in zip(df["Metal"], df["Facet"])
    ])

    y = df["Binding_Energy_scaled"].values

    return X, y

# ================= MODEL =================
def build_model(input_dim, units, dropout):

    model = keras.Sequential([
        layers.Dense(units, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(units, activation="relu"),
        layers.Dense(1)
    ])

    return model


# ================= MAIN =================
if __name__=="__main__":

    df = pd.read_csv("../../../../../output_with_smiles.csv")

    print("Dataset size:", df.shape)
    df = df[(df["Binding_Energy_eV"] >= -10) & (df["Binding_Energy_eV"] <= 10)]
    print("Dataset size:", df.shape)

    scaler = StandardScaler()
    df["Binding_Energy_scaled"] = scaler.fit_transform(df[["Binding_Energy_eV"]])

    # ===== SPLIT =====
    df_train = df.sample(frac=0.8, random_state=42)
    df_rem   = df.drop(df_train.index)
    df_val   = df_rem.sample(frac=0.5, random_state=42)
    df_test  = df_rem.drop(df_val.index)

    X_train, y_train = create_dataset(df_train)
    X_val, y_val     = create_dataset(df_val)
    X_test, y_test   = create_dataset(df_test)

    # ===== PARAM GRID (TUNING ONLY) =====
    param_grid = [
        {"units":32, "dropout":0.2, "lr":1e-3},
        {"units":64, "dropout":0.2, "lr":1e-4},
        # {"units":64, "dropout":0.3, "lr":1e-4},
        # {"units":128,"dropout":0.3, "lr":5e-5},
    ]

    best_model = None
    best_loss  = float("inf")
    best_history = None

    for params in param_grid:
        print("\nTesting:", params)

        model = build_model(
            X_train.shape[1],
            params["units"],
            params["dropout"]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(params["lr"]),
            loss="mse",
            metrics=["mae"]
        )

        early = keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[early]
        )

        val_loss = min(history.history["val_loss"])
        print("Val Loss:", val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_history = history

    print("\nBest Val Loss:", best_loss)

    model = best_model

    # ===== TEST =====
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

    print("\nMAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    # ===== PLOTS =====
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(best_history.history["loss"], label="Train")
    plt.plot(best_history.history["val_loss"], label="Val")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test),max(y_test)],
             [min(y_test),max(y_test)], "--")
    plt.title(f"Parity Plot (R2={r2_score(y_test,y_pred):.2f})")

    plt.show()