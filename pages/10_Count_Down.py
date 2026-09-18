from __future__ import annotations
from datetime import datetime, timezone
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Count Down", page_icon="⏱️", layout="wide")

DEFAULT_EVENTS = [
    "dock off", "sailing window open", "sailing closed before fly by",
    "fly by for FRA", "sailing reopen after fly by", "start first Race for FRA",
    "start first race for other group", "start second Race for FRA",
    "start second Race for other group", "start third Race for FRA",
    "start third Race for other group", "start Final Race",
    "sailing window is closed", "custom 1", "custom 2", "custom 3",
]

if "countdown_utc_offset" not in st.session_state:
    st.session_state.countdown_utc_offset = 2
if "countdown_events" not in st.session_state:
    st.session_state.countdown_events = pd.DataFrame({
        "Afficher": [False] * len(DEFAULT_EVENTS),
        "Nom": DEFAULT_EVENTS,
        "Heure locale": [0] * len(DEFAULT_EVENTS),
        "Minute": [0] * len(DEFAULT_EVENTS),
    })

def render_clocks(offset_h: int):
    base = datetime.now(timezone.utc).timestamp() * 1000
    components.html(f'''<!doctype html><html><style>
    html,body{{margin:0;background:transparent;font-family:Arial,sans-serif;overflow:hidden}}
    .row{{display:flex;gap:18px}}.box{{flex:1;border:1px solid #8886;border-radius:10px;padding:12px 18px}}
    .lab{{font-size:15px;opacity:.65}}.tm{{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums}}
    </style><body><div class="row"><div class="box"><div class="lab">Heure UTC</div><div id="u" class="tm"></div></div>
    <div class="box"><div class="lab">Heure locale (UTC{offset_h:+d})</div><div id="l" class="tm"></div></div></div>
    <script>(()=>{{const b={base:.3f},p=performance.now(),o={offset_h}*3600000;
    function f(x){{let d=new Date(x);return String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0')+':'+String(d.getUTCSeconds()).padStart(2,'0')}}
    function t(){{let n=b+performance.now()-p;u.textContent=f(n);l.textContent=f(n+o)}}t();setInterval(t,100)}})();</script></body></html>''', height=92, scrolling=False)

def render_countdowns(df: pd.DataFrame, offset_h: int):
    a = df.loc[df["Afficher"].fillna(False).astype(bool)].copy()
    if a.empty:
        st.info("Coche les événements à afficher pour faire apparaître les comptes à rebours.")
        return
    a["Heure locale"] = pd.to_numeric(a["Heure locale"], errors="coerce").fillna(0).clip(0,23).astype(int)
    a["Minute"] = pd.to_numeric(a["Minute"], errors="coerce").fillna(0).clip(0,59).astype(int)
    a["Nom"] = a["Nom"].fillna("").astype(str)
    a["_sort"] = a["Heure locale"]*60+a["Minute"]
    a = a.sort_values("_sort", kind="stable").reset_index(drop=True)
    ev = [{"name":r["Nom"],"hour":int(r["Heure locale"]),"minute":int(r["Minute"]),
           "special":("FRA" in r["Nom"].upper()) or r["Nom"].strip().lower()=="start final race"} for _,r in a.iterrows()]
    payload=json.dumps(ev,ensure_ascii=False)
    base=datetime.now(timezone.utc).timestamp()*1000
    components.html(f'''<!doctype html><html><style>
    html,body{{margin:0;background:transparent;font-family:Arial,sans-serif}}.e{{display:grid;grid-template-columns:minmax(280px,1.7fr) 120px minmax(180px,.8fr);gap:16px;align-items:center;padding:11px 14px;border-bottom:1px solid #8884;font-size:20px}}
    .e:first-child{{font-weight:800;border-top:1px solid #8884}}.special{{color:#e32636}}.when{{opacity:.72;font-variant-numeric:tabular-nums}}.cd{{text-align:right;font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}}.e:first-child .cd{{font-weight:900}}
    </style><body><div id="list"></div><script>(()=>{{const E={payload},off={offset_h},b={base:.3f},p=performance.now(),L=document.getElementById('list');
    const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
    E.forEach((e,i)=>{{let r=document.createElement('div');r.className='e';r.innerHTML='<div class="'+(e.special?'special':'')+'">'+esc(e.name)+'</div><div class="when">'+String(e.hour).padStart(2,'0')+':'+String(e.minute).padStart(2,'0')+'</div><div class="cd" id="c'+i+'"></div>';L.appendChild(r)}});
    function target(e,n){{let d=new Date(n+off*3600000);return Date.UTC(d.getUTCFullYear(),d.getUTCMonth(),d.getUTCDate(),e.hour,e.minute)-off*3600000}}
    function fmt(ms){{let neg=ms<0,s=Math.floor(Math.abs(ms)/1000),h=Math.floor(s/3600);s-=h*3600;let m=Math.floor(s/60),q=s-m*60;return(neg?'−':'')+String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(q).padStart(2,'0')}}
    function tick(){{let n=b+performance.now()-p;E.forEach((e,i)=>document.getElementById('c'+i).textContent=fmt(target(e,n)-n))}}tick();setInterval(tick,100)}})();</script></body></html>''', height=max(90,55*len(ev)+15), scrolling=False)

st.title("Count Down")
c1,c2=st.columns([4.6,1.2],vertical_alignment="center")
with c1: render_clocks(int(st.session_state.countdown_utc_offset))
with c2:
    st.caption("Offset heure locale / UTC")
    o=st.number_input("UTC offset (h)",-12,14,int(st.session_state.countdown_utc_offset),1,label_visibility="collapsed")
    if int(o)!=int(st.session_state.countdown_utc_offset):
        st.session_state.countdown_utc_offset=int(o);st.rerun()
    st.write(f"**UTC{int(st.session_state.countdown_utc_offset):+d}**")

st.divider();st.subheader("Événements")
edited=st.data_editor(st.session_state.countdown_events,use_container_width=True,hide_index=True,num_rows="fixed",
    column_config={
      "Afficher":st.column_config.CheckboxColumn("Afficher",default=False,width="small"),
      "Nom":st.column_config.TextColumn("Nom",width="large"),
      "Heure locale":st.column_config.NumberColumn("Heure locale",min_value=0,max_value=23,step=1,format="%d",width="small"),
      "Minute":st.column_config.NumberColumn("Minute",min_value=0,max_value=59,step=1,format="%d",width="small")},
    key="countdown_events_editor")
st.session_state.countdown_events=edited.copy()
st.divider();st.subheader("Comptes à rebours")
render_countdowns(edited,int(st.session_state.countdown_utc_offset))
