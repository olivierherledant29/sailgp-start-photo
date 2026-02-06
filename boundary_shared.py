import pandas as pd
import streamlit as st

from xml_boundary import parse_course_limit_xml, parse_marks_xml


def sidebar_boundary_uploader() -> pd.DataFrame:
    st.markdown(
        "<div style='padding:8px;border-radius:8px;"
        "border:1px solid #7B1FA2;background:#F3E5F5;'>"
        "<b>XML commun – Boundary</b></div>",
        unsafe_allow_html=True,
    )

    xml_file = st.file_uploader(
        "Boundary XML (CourseLimit name='Boundary')",
        type=["xml"],
        accept_multiple_files=False,
    )

    if xml_file is None:
        st.session_state.pop("marks_df", None)
        st.session_state.pop("boundary_xml_name", None)
        return pd.DataFrame(columns=["ring", "seq", "lat", "lon"])

    # ✅ Stocker le nom pour les pages qui veulent en déduire un TWD
    st.session_state["boundary_xml_name"] = getattr(xml_file, "name", None)

    xml_bytes = xml_file.getvalue()

    boundary_df = parse_course_limit_xml(xml_bytes)

    try:
        marks_df = parse_marks_xml(xml_bytes)
    except Exception:
        marks_df = pd.DataFrame(columns=["mark", "lat", "lon"])
    st.session_state["marks_df"] = marks_df

    return boundary_df
