"""Optional, fixed Arabic translations shown below the English explanations.

No external translation API or user-provided text is transmitted.
"""
from html import escape
import streamlit as st


def arabic_below(arabic_text):
    """Display an Arabic translation below nearby English explanatory text."""
    if st.session_state.get("show_arabic", False):
        st.markdown(
            '<div dir="rtl" lang="ar" style="text-align:right;'
            'border-right:3px solid #18aaa7;padding:7px 12px;'
            'margin:4px 0 15px 0;color:inherit;line-height:1.9;'
            'font-size:0.95rem;">'
            + escape(str(arabic_text))
            + '</div>',
            unsafe_allow_html=True,
        )


def explanation(english_text, arabic_text, kind="caption"):
    """English by default; Arabic appears immediately beneath when enabled."""
    if kind=="info":
        st.info(english_text)
    elif kind=="write":
        st.write(english_text)
    elif kind=="markdown":
        st.markdown(english_text)
    else:
        st.caption(english_text)
    arabic_below(arabic_text)
