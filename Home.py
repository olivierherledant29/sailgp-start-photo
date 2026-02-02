import streamlit as st

st.set_page_config(page_title="SailGP Start Photo", layout="wide")
st.title("SailGP – Start Photo")

st.write("Choisir le mode :")
c1, c2, c3 = st.columns(3)

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

with c3:
    st.subheader("start_aid only")
    st.write("ouvrir start_aid, lit seulement xml")
    if st.button("Ouvrir start_aid"):
        st.switch_page("pages/3_Start_Aid.py")
