import pandas as pd
import streamlit as st
from xml_boundary import parse_course_limit_xml


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
        return pd.DataFrame(columns=["ring", "seq", "lat", "lon"])

    return parse_course_limit_xml(xml_file.getvalue())
