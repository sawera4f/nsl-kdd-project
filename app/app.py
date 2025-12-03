import streamlit as st
import pickle
import numpy as np

st.title("🔍 Détection d’attaques réseau – NSL-KDD")
st.write("Interface interactive simplifiée basée sur un modèle réduit (7 features).")

# -----------------------------------------------------
# Charger le modèle réduit + encoders
# -----------------------------------------------------
try:
    model = pickle.load(open("models/logreg_small.pkl", "rb"))
    encoders = pickle.load(open("models/encoders.pkl", "rb"))
    st.success("Modèle et encoders chargés.")
except Exception as e:
    st.error(f"Erreur chargement modèle : {e}")
    st.stop()

# -----------------------------------------------------
# 📝 Zone interactive
# -----------------------------------------------------
st.header("📝 Tester une connexion")

# Les noms EXACTS des colonnes de ton dataset :
# "0" = duration
# "491" = src_bytes
# "0.1" = dst_bytes
# "0.20" = count
# "tcp" = protocole
# "ftp_data" = service
# "SF" = flag

duration = st.number_input("Durée (colonne '0')", min_value=0.0)
src_bytes = st.number_input("Bytes envoyés (colonne '491')", min_value=0.0)
dst_bytes = st.number_input("Bytes reçus (colonne '0.1')", min_value=0.0)
count = st.number_input("Count (colonne '0.20')", min_value=0)

protocol = st.selectbox("Protocole (colonne 'tcp')", sorted(encoders["tcp"].classes_))
service = st.selectbox("Service (colonne 'ftp_data')", sorted(encoders["ftp_data"].classes_))
flag = st.selectbox("Flag (colonne 'SF')", sorted(encoders["SF"].classes_))

# -----------------------------------------------------
# Prédiction
# -----------------------------------------------------
if st.button("🔍 Prédire"):
    try:
        proto_enc = encoders["tcp"].transform([protocol])[0]
        service_enc = encoders["ftp_data"].transform([service])[0]
        flag_enc = encoders["SF"].transform([flag])[0]

        X = np.array([[
            duration,      # "0"
            proto_enc,     # "tcp"
            service_enc,   # "ftp_data"
            flag_enc,      # "SF"
            src_bytes,     # "491"
            dst_bytes,     # "0.1"
            count          # "0.20"
        ]])

        pred = model.predict(X)[0]

        if pred == 1:
            st.error("⚠️ Résultat : **ATTAQUE**")
        else:
            st.success("✔️ Résultat : **NORMAL**")

    except Exception as e:
        st.error(f"Erreur : {e}")
