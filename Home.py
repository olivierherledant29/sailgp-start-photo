import streamlit as st

st.set_page_config(page_title="SailGP Start Photo", layout="wide")
st.title("SailGP – Start Photo")

st.write("Choisir le mode :")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Replay")
    st.write("Post-analyse : choix d’un départ, offset, traces, crossing, boundary XML…")
    if st.button("Ouvrir Replay"):
        st.switch_page("pages/1_Replay.py")

with c2:
    st.subheader("Live")
    st.write("Calé sur l’heure UTC actuelle. (Le live peut être vide hors navigation.)")
    if st.button("Ouvrir Live"):
        st.switch_page("pages/2_Live.py")
