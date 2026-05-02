#this modeli is only take molecules as input 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
np.random.seed(42)
tf.random.set_seed(42)

# ================= FEATURES =================
def encode_atom(atom):
    syms = ["B","Br","C","Ca","Cl","F","H","I","N","Na","O","P","S"]
    atom_type = [1.0 if atom.GetSymbol()==s else 0.0 for s in syms]

    hybrid_types = ["s","sp","sp2","sp3","sp3d","sp3d2"]
    hybrid = [1.0 if atom.GetHybridization().name.lower()==h else 0.0 for h in hybrid_types]

    extra = [
        atom.GetDegree()/6,
        atom.GetFormalCharge()/3,
        atom.GetTotalNumHs()/4,
        float(atom.GetIsAromatic())
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
    atom_feats,bond_feats,pair_indices = x_batch

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
        mol_indicator
    ), y_batch

def create_dataset(df,batch_size=16,shuffle=False):
    graphs=[build_graph(s) for s in df["smiles"]]

    a=tf.ragged.constant([g[0] for g in graphs],dtype=tf.float32)
    b=tf.ragged.constant([g[1] for g in graphs],dtype=tf.float32)
    p=tf.ragged.constant([g[2] for g in graphs],dtype=tf.int64)
    y=tf.constant(df["Binding_Energy_scaled"].values,dtype=tf.float32)

    ds=tf.data.Dataset.from_tensor_slices(((a,b,p),y))

    if shuffle:
        ds=ds.shuffle(len(df))

    return ds.batch(batch_size).map(prepare_batch).prefetch(tf.data.AUTOTUNE)

# ================= MODEL =================
class EdgeNetwork(layers.Layer):
    def build(self,input_shape):
        self.atom_dim=input_shape[0][-1]
        self.bond_dim=input_shape[1][-1]

        self.kernel=self.add_weight(
            shape=(self.bond_dim,self.atom_dim**2),
            initializer="glorot_uniform",
    
        )
        self.bias=self.add_weight(shape=(self.atom_dim**2,),initializer="zeros")

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

def build_model(a_dim,b_dim,units,steps,dropout):
    in_a=layers.Input((a_dim,))
    in_b=layers.Input((b_dim,))
    in_p=layers.Input((2,),dtype="int32")
    in_i=layers.Input((),dtype="int32")

    x = MessagePassing(units,steps)([in_a,in_b,in_p])
    pool=PoolingLayer()([x,in_i])

    x = layers.Dense(units,activation="relu", kernel_regularizer=regularizers.l1(1e-4))(pool)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units,activation="relu")(x)

    out = layers.Dense(1)(x)

    return keras.Model(inputs=[in_a,in_b,in_p,in_i],outputs=out)

# ================= MAIN =================
if __name__=="__main__":

    df = pd.read_csv("output_with_smiles.csv")
    df = df.dropna(subset=["smiles"])
    df["smiles"] = df["smiles"].astype(str)

    # ===== OUTLIER REMOVE =====
    df = df[(df["Binding_Energy_eV"] >= -10) & (df["Binding_Energy_eV"] <= 10)]
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

    param_grid = [
        {"units":32,"steps":3,"lr":2e-4,"dropout":0.2},
        {"units":32,"steps":5,"lr":1e-4,"dropout":0.3},
        {"units":64,"steps":5,"lr":1e-4,"dropout":0.3},
    ]

    best_model=None
    best_loss=float("inf")
    best_history=None

    for params in param_grid:
        print("\nTesting:", params)

        model=build_model(
            a_dim,b_dim,
            params["units"],
            params["steps"],
            params["dropout"]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(params["lr"]),
            loss="mse",
            metrics=["mae"]
        )

        early=keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        )

        hist=model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            verbose=0,
            callbacks=[early]
        )

        val_loss=min(hist.history["val_loss"])
        print("Val Loss:", val_loss)

        if val_loss<best_loss:
            best_loss=val_loss
            best_model=model
            best_history=hist

    model=best_model

    # ===== TEST =====
    y_pred_scaled=model.predict(test_ds).flatten()

    y_pred=scaler.inverse_transform(
        y_pred_scaled.reshape(-1,1)
    ).flatten()

    y_true=df_test["Binding_Energy_eV"].values

    print("\nMAE:",mean_absolute_error(y_true,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(y_true,y_pred)))
    print("R2:",r2_score(y_true,y_pred))

    # ===== PLOTS =====
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(best_history.history["loss"], label="Train")
    plt.plot(best_history.history["val_loss"], label="Val")
    plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(y_true,y_pred)
    plt.plot([min(y_true),max(y_true)],
             [min(y_true),max(y_true)],
             "--")
    plt.title(f"Parity Plot (R2={r2_score(y_true,y_pred):.2f})")

    plt.show()