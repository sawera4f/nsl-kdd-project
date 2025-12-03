import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Charger le dataset
df = pd.read_csv("data/raw/NSL_KDD_Train.csv")

# Colonnes choisies (adaptées à ton CSV)
cols = [
    "0",        # duration
    "tcp",      # protocol_type (texte → encoder)
    "ftp_data", # service (texte → encoder)
    "SF",       # flag (texte → encoder)
    "491",      # src_bytes
    "0.1",      # dst_bytes
    "0.20"      # count
]

df_small = df[cols + ["normal"]].copy()

# -----------------------------------------
# Encoder les colonnes texte
# -----------------------------------------
text_cols = ["tcp", "ftp_data", "SF"]

encoder_dict = {}

for col in text_cols:
    le = LabelEncoder()
    df_small[col] = le.fit_transform(df_small[col].astype(str))
    encoder_dict[col] = le

# -----------------------------------------
# Encoder la cible (normal / attaque)
# -----------------------------------------
df_small["normal"] = (df_small["normal"] != "normal").astype(int)

# X et y
X = df_small[cols]
y = df_small["normal"]

# -----------------------------------------
# Train/test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------------------
# Modèle
# -----------------------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# -----------------------------------------
# Sauvegarde du modèle et des encoders
# -----------------------------------------
pickle.dump(model, open("models/logreg_small.pkl", "wb"))
pickle.dump(encoder_dict, open("models/encoders.pkl", "wb"))

print("Modèle réduit + encoders sauvegardés !")
