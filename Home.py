import streamlit as st

st.set_page_config(page_title="SailGP Start Photo", layout="wide")
st.title("SailGP – Start Photo")

st.write("Choisir le mode :")
c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)

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
    st.write("ouvrir start_aid, lit seulement xml et polaire")
    if st.button("Ouvrir start_aid"):
        st.switch_page("pages/3_Start_Aid.py")

with c4:
    st.subheader("routeur")
    st.write("ouvrir routeur, lit seulement xml et polaire")
    if st.button("Ouvrir routeur"):
        st.switch_page("pages/4_Routeur.py")

with c5:
    st.subheader("board cycles count")
    st.write("Compteur babord/tribord (manuel) + POIs API (beta).")
    if st.button("Ouvrir board cycles count"):
        st.switch_page("pages/5_Board_Cycles_Count.py")

with c6:
    st.subheader("Jib Trim")
    st.write("Analyse foc : lead, sheet, cunno, leeway, VMG target.")
    if st.button("Ouvrir Jib Trim"):
        st.switch_page("pages/6_Jib_Trim.py")

with c7:
    st.subheader("FC Data")
    st.write("Analyse flight control : cant, ride height, rudder, BSP/TWA.")
    if st.button("Ouvrir FC Data"):
        st.switch_page("pages/7_FC_Data.py")

with c8:
    st.subheader("Wing Trim")
    st.write("Analyse aile : CA1, twist, targets interpolées.")
    if st.button("Ouvrir Wing Trim"):
        st.switch_page("pages/8_Wing_Trim.py")

with c9:
    st.subheader("Winning Traces")
    st.write("recherche meilleures traces sur une course")
    if st.button("Ouvrir Winning traces"):
        st.switch_page("pages/9_Winning_Traces.py")