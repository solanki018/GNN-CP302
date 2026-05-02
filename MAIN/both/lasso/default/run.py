#the model is for our full dataset, including both surface and molecule features. It uses a GNN for the molecule and concatenates it with surface features before final prediction.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from mendeleev import element
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
import re

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
np.random.seed(42)
tf.random.set_seed(42)

# ================= PROPERTY CACHE =================
PROPERTY_CACHE = {}

D_BAND_FILLING = {"Ag":1.0,"Au":1.0,"Cd":1.0,"Co":0.8,"Cu":1.0,"Fe":0.7,"Ir":0.8,
"Ni":0.9,"Os":0.7,"Pd":0.9,"Pt":0.9,"Rh":0.8,"Ru":0.7,"Zn":1.0}

ENTHALPY_FUSION = {"Ag":105,"Au":64,"Cd":112,"Co":272,"Cu":205,"Fe":247,"Ir":266,
"Ni":290,"Os":134,"Pd":157,"Pt":113,"Rh":258,"Ru":381,"Zn":112}

D_BAND_CENTER = {"Ag":-4.7,"Au":-3.6,"Cd":-6.2,"Co":-1.6,"Cu":-2.7,"Fe":-1.8,"Ir":-2.5,
"Ni":-1.4,"Os":-2.5,"Pd":-1.8,"Pt":-2.2,"Rh":-1.7,"Ru":-2.4,"Zn":-6.0}

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

# ================= FEATURES =================
def encode_atom(atom):
    syms = ["B","Br","C","Ca","Cl","F","H","I","N","Na","O","P","S"]
    atom_type = [1.0 if atom.GetSymbol()==s else 0.0 for s in syms]

    hybrid_types = ["s","sp","sp2","sp3","sp3d","sp3d2"]
    hybrid = [1.0 if atom.GetHybridization().name.lower()==h else 0.0 for h in hybrid_types]

    props = get_mendeleev_data(atom.GetSymbol())

    extra = [
        atom.GetDegree()/6,
        atom.GetFormalCharge()/3,
        atom.GetTotalNumHs()/4,
        float(atom.GetIsAromatic()),
        props["en"],
        props["group"]
    ]

    return np.concatenate([atom_type, hybrid, extra]).astype(np.float32)

def encode_bond(bond):
    if bond is None:
        return np.zeros(6,dtype=np.float32)
    return np.array([
        bond.GetBondType()==Chem.rdchem.BondType.SINGLE,
        bond.GetBondType()==Chem.rdchem.BondType.DOUBLE,
        bond.GetBondType()==Chem.rdchem.BondType.TRIPLE,
        bond.GetBondType()==Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ],dtype=np.float32)

def get_surface_vector(metal, facet):
    p = get_mendeleev_data(metal)
    m_feat = [p["group"],p["en"],p["ie"],p["ea"],p["mp"],p["bp"],
              p["d_band"],p["hfus"],p["wf"],p["cohesive"]]

    facet_digits = "".join(re.findall(r'\d', str(facet)))
    f_str = facet_digits.ljust(4,'0')
    f_feat = [float(d)/9.0 for d in f_str[:4]]

    return np.array(m_feat + f_feat, dtype=np.float32)

# ================= GRAPH =================
def build_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    atoms,bonds,pairs=[],[],[]

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atoms.append(encode_atom(atom))

        pairs.append([idx,idx])
        bonds.append(encode_bond(None))

        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(idx,nb.GetIdx())
            pairs.append([idx,nb.GetIdx()])
            bonds.append(encode_bond(bond))

    return np.array(atoms),np.array(bonds),np.array(pairs)

# ================= DATA =================
def prepare_batch(x_batch,y_batch):
    atom_feats,bond_feats,pair_indices,surf_vectors = x_batch

    num_atoms = atom_feats.row_lengths()
    num_bonds = bond_feats.row_lengths()

    mol_indices = tf.range(tf.shape(num_atoms)[0])
    mol_indicator = tf.repeat(mol_indices,num_atoms)

    increment = tf.pad(tf.cumsum(num_atoms[:-1]),[[1,0]])
    increment_per_bond = tf.gather(increment,tf.repeat(mol_indices,num_bonds))

    flat_pairs = pair_indices.merge_dims(0,1).to_tensor()
    flat_pairs = flat_pairs + tf.cast(increment_per_bond[:,None],tf.int64)

    return (
        atom_feats.merge_dims(0,1).to_tensor(),
        bond_feats.merge_dims(0,1).to_tensor(),
        flat_pairs,
        surf_vectors,
        mol_indicator
    ), y_batch

def create_dataset(df,batch_size=16,shuffle=False):
    graphs=[build_graph(s) for s in df["smiles"]]
    surfaces=[get_surface_vector(m,f) for m,f in zip(df["Metal"],df["Facet"])]

    a=tf.ragged.constant([g[0] for g in graphs],dtype=tf.float32)
    b=tf.ragged.constant([g[1] for g in graphs],dtype=tf.float32)
    p=tf.ragged.constant([g[2] for g in graphs],dtype=tf.int64)
    s=tf.constant(surfaces,dtype=tf.float32)
    y=tf.constant(df["Binding_Energy_scaled"].values,dtype=tf.float32)

    ds=tf.data.Dataset.from_tensor_slices(((a,b,p,s),y))
    if shuffle:
        ds=ds.shuffle(len(df))

    return ds.batch(batch_size).map(prepare_batch).prefetch(tf.data.AUTOTUNE)

# ================= MODEL =================
class EdgeNetwork(layers.Layer):
    def build(self,input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]

        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim**2),
            initializer="glorot_uniform"
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim**2,),
            initializer="zeros"
        )

    def call(self,inputs):
        atoms,bonds,pairs = inputs

        gate=tf.reshape(tf.matmul(bonds,self.kernel)+self.bias,
                        (-1,self.atom_dim,self.atom_dim))

        neighbors=tf.expand_dims(tf.gather(atoms,pairs[:,1]),-1)
        msg=tf.squeeze(tf.matmul(gate,neighbors),-1)

        return tf.math.unsorted_segment_sum(msg,pairs[:,0],
                                            num_segments=tf.shape(atoms)[0])

class MessagePassing(layers.Layer):
    def __init__(self,units=32,steps=5):
        super().__init__()
        self.proj=layers.Dense(units,activation="relu")
        self.edge_net=EdgeNetwork()
        self.gru=layers.GRUCell(units)
        self.steps=steps

    def call(self,inputs):
        atoms,bonds,pairs=inputs
        x=self.proj(atoms)

        for _ in range(self.steps):
            m=self.edge_net([x,bonds,pairs])
            x,_=self.gru(m,x)

        return x

class PoolingLayer(layers.Layer):
    def call(self,inputs):
        return tf.math.segment_mean(inputs[0],inputs[1])

def build_model(a_dim,b_dim,s_dim,units,steps,dropout):

    in_a=layers.Input((a_dim,))
    in_b=layers.Input((b_dim,))
    in_p=layers.Input((2,),dtype="int32")
    in_s=layers.Input((s_dim,))
    in_i=layers.Input((),dtype="int32")

    x = MessagePassing(units,steps)([in_a,in_b,in_p])
    pool=PoolingLayer()([x,in_i])
    comb=layers.Concatenate()([pool,in_s])

    x = layers.Dense(units,activation="relu",
                     kernel_regularizer=regularizers.l1(1e-4))(comb)

    x = layers.Dropout(dropout)(x)

    x = layers.Dense(units,activation="relu",
                     kernel_regularizer=regularizers.l1(1e-4))(x)

    out = layers.Dense(1,
                       kernel_regularizer=regularizers.l1(1e-4))(x)

    return keras.Model(inputs=[in_a,in_b,in_p,in_s,in_i],outputs=out)

# ================= MODEL =================



# ================= MAIN =================
if __name__=="__main__":

    df = pd.read_csv("../../../../../output_with_smiles.csv")
    df = df.dropna(subset=["smiles"])
    df["smiles"] = df["smiles"].astype(str)

    # df = df[(df["Binding_Energy_eV"] >= -10) & (df["Binding_Energy_eV"] <= 10)]
    print("After cleaning:", df.shape)

    scaler = StandardScaler()
    df["Binding_Energy_scaled"] = scaler.fit_transform(df[["Binding_Energy_eV"]])

    # ===== SPLIT =====
    df_train = df.sample(frac=0.8, random_state=42)
    df_rem   = df.drop(df_train.index)
    df_val   = df_rem.sample(frac=0.5, random_state=42)
    df_test  = df_rem.drop(df_val.index)

    train_ds = create_dataset(df_train,shuffle=True)
    val_ds   = create_dataset(df_val)
    test_ds  = create_dataset(df_test)

    sample_g = build_graph(df["smiles"].iloc[0])
    a_dim = sample_g[0].shape[-1]
    b_dim = sample_g[1].shape[-1]
    s_dim = len(get_surface_vector(df["Metal"].iloc[0], df["Facet"].iloc[0]))

   # ===== FIXED MODEL (WITH RIDGE, NO TUNING) =====
    model = build_model(
        a_dim, b_dim, s_dim,
        units=64,
        steps=5,
        dropout=0.2
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"]
    )

    early = keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        verbose=1,
        callbacks=[early]
    )

    # ===== TEST =====
    y_pred_scaled = model.predict(test_ds).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = df_test["Binding_Energy_eV"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\nMAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    # ===== PLOTS =====
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(y_true,y_pred)
    plt.plot([min(y_true),max(y_true)],
             [min(y_true),max(y_true)],
             "--")
    plt.title(f"Parity Plot (R2={r2:.2f})")

    plt.show()