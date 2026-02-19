import math
from datetime import date
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
st.set_page_config(page_title="KADO94", layout="wide")

# -----------------------------
# -----------------------------
# 入力：数字だけ打てばOK → 自動でカンマ整形（例: 1039071652 → 1,039,071,652）
#
# Streamlitのnumber_inputは入力欄にカンマ表示ができないため、text_inputで実現します。
# ただし、text_inputの「自分自身の値」をコールバック内で書き換えると反映されない環境があるため、
# **ウィジェットkeyを世代管理して描画し直す**方式で、確実に入力欄をカンマ表示へ更新します。
# -----------------------------
def _clean_num_text(s: str) -> str:
    return (s or "").replace(",", "").replace(" ", "").replace("　", "").replace("_", "")

def _rerun():
    # Streamlitのバージョン差分吸収
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def int_input_comma(label: str, key: str, default: int = 0, help: str | None = None) -> int:
    """整数入力（数字だけでOK）。確定後に入力欄をカンマ整形して表示する。"""
    fmt_key = f"{key}__fmt"
    num_key = f"{key}__num"
    rev_key = f"{key}__rev"

    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0
        st.session_state[fmt_key] = f"{int(default):,}"
        st.session_state[num_key] = int(default)

    widget_key = f"{key}__w{st.session_state[rev_key]}"
    txt = st.text_input(
        label,
        value=st.session_state[fmt_key],
        key=widget_key,
        help=help or "数字だけ入力でOK（例: 1039071652）",
    )

    cleaned = _clean_num_text(txt)
    if cleaned == "":
        num = int(default)
    else:
        sign = -1 if cleaned.startswith("-") else 1
        num_part = cleaned[1:] if cleaned.startswith("-") else cleaned
        if not num_part.isdigit():
            st.error(f"「{label}」は数字で入力してね（例: 123456）。")
            return int(st.session_state[num_key])
        num = sign * int(num_part)

    formatted = f"{num:,}"
    # 表示用テキストが変わったら、ウィジェットを世代交代して再描画（入力欄にカンマ反映）
    if formatted != st.session_state[fmt_key]:
        st.session_state[fmt_key] = formatted
        st.session_state[num_key] = num
        st.session_state[rev_key] += 1
        _rerun()

    st.session_state[num_key] = num
    return num

def float_input_comma(label: str, key: str, default: float = 0.0, digits: int = 2, help: str | None = None) -> float:
    """小数入力（数字だけでOK）。確定後に入力欄をカンマ整形して表示する。"""
    fmt_key = f"{key}__fmt"
    num_key = f"{key}__num"
    rev_key = f"{key}__rev"

    if rev_key not in st.session_state:
        st.session_state[rev_key] = 0
        st.session_state[fmt_key] = f"{float(default):,.{digits}f}"
        st.session_state[num_key] = float(default)

    widget_key = f"{key}__w{st.session_state[rev_key]}"
    txt = st.text_input(
        label,
        value=st.session_state[fmt_key],
        key=widget_key,
        help=help or "数字だけ入力でOK（例: 85911.25）",
    )

    cleaned = _clean_num_text(txt)
    if cleaned == "":
        num = float(default)
    else:
        try:
            num = float(cleaned)
        except Exception:
            st.error(f"「{label}」は数字で入力してね（例: 12345.67）。")
            return float(st.session_state[num_key])

    formatted = f"{num:,.{digits}f}"
    if formatted != st.session_state[fmt_key]:
        st.session_state[fmt_key] = formatted
        st.session_state[num_key] = num
        st.session_state[rev_key] += 1
        _rerun()

    st.session_state[num_key] = num
    return num



def _chart_settings(size: str):
    """グラフの幅/高さを調整（横幅が長すぎ問題対策）"""
    if size == "コンパクト":
        return {"w_main": 640, "w_period": 720, "h_occ": 95, "h_bar": 220, "h_need": 170, "h_line": 260, "use_container": False}
    if size == "標準":
        return {"w_main": 760, "w_period": 900, "h_occ": 105, "h_bar": 250, "h_need": 190, "h_line": 290, "use_container": False}
    # ワイド（全幅）
    return {"w_main": None, "w_period": None, "h_occ": 115, "h_bar": 260, "h_need": 200, "h_line": 300, "use_container": True}

def _plotly_center(fig, width: int | None, use_container: bool):
    if use_container:
        st.plotly_chart(fig, use_container_width=True)
        return
    cols = st.columns([1, 8, 1])
    with cols[1]:
        st.plotly_chart(fig, use_container_width=False)


def apply_rounding(mode: str, **vals):
    """
    mode:
      - "高精度": 途中の丸めなし（理論値に近い）
      - "Excel互換": 途中計算を円・人日単位で丸め（Excelの表示結果に合わせやすい）
    """
    if mode != "Excel互換":
        return vals

    out = dict(vals)

    def r0(x):
        try:
            return float(round(float(x)))
        except Exception:
            return x

    # 円（収入・単価）は1円単位、人日は1人日単位で丸め
    for k in ["unit_price", "revenue_actual", "revenue_target", "delta_revenue"]:
        if k in out and out[k] is not None:
            out[k] = r0(out[k])

    for k in ["required_patient_days_target", "max_patient_days_100", "add_patient_days", "patient_days_actual"]:
        if k in out and out[k] is not None:
            out[k] = r0(out[k])

    return out
def simulate_month(
    month_start: pd.Timestamp,
    target_occ: float,
    beds: float,
    los: float,
    unit_price: float,  # 入院単価（円/人日）
    # 目標の指定方法（目標値だけシミュレーションで使用）
    target_mode: str = "稼働率",
    target_admissions: float | None = None,
    target_revenue: float | None = None,
    # 実績（任意）
    patient_days_actual: float | None = None,
    admissions_actual: float | None = None,
    revenue_actual: float | None = None,
):
    """月次シミュレーション。

    - 目標は基本「稼働率（target_occ）」で計算
    - ただし target_mode が「新入院数」「入院収入」の場合は、それを起点に必要延べ患者数を逆算
    - 実績は任意入力。未入力なら比較系（増収額・追加必要量）は None を返す
    """
    d = days_in_month(month_start)
    max_patient_days = beds * d

    # --- 目標の置き方 ---
    required_patient_days = None
    if target_mode == "新入院数" and target_admissions is not None and los and los > 0:
        # 新入院数（目標）→ 必要延べ患者数（人日）
        required_patient_days = float(target_admissions) * float(los)
    elif target_mode == "入院収入" and target_revenue is not None and unit_price and unit_price > 0:
        # 入院収入（目標）→ 必要延べ患者数（人日）
        required_patient_days = float(target_revenue) / float(unit_price)
    else:
        # 稼働率（目標）→ 必要延べ患者数（人日）
        required_patient_days = float(max_patient_days) * float(target_occ)

    # 計算上の目標稼働率（新入院数/入院収入起点のときに表示用）
    target_occ_calc = (required_patient_days / max_patient_days) if max_patient_days else None

    # 目標：必要新入院（目標だけでも出せる）
    if target_mode == "新入院数" and target_admissions is not None:
        required_admissions_target = math.ceil(float(target_admissions))
    else:
        required_admissions_target = math.ceil(required_patient_days / los) if los and los > 0 else None

    # --- 実績があれば比較系を計算 ---
    occ_actual = None
    add_patient_days = None
    add_admissions = None
    required_admissions = None
    delta_revenue = None

    if patient_days_actual is not None and max_patient_days:
        occ_actual = (patient_days_actual / max_patient_days)

    revenue_target = required_patient_days * unit_price

    if revenue_actual is not None:
        delta_revenue = revenue_target - revenue_actual

    if patient_days_actual is not None:
        add_patient_days = max(0.0, required_patient_days - patient_days_actual)

        if los and los > 0:
            add_admissions = math.ceil(add_patient_days / los)

        if admissions_actual is not None and add_admissions is not None:
            required_admissions = admissions_actual + add_admissions

    return {
        "month_start": month_start,
        "month_days": d,
        "max_patient_days_100": max_patient_days,
        "required_patient_days_target": required_patient_days,
        "target_occ_calc": target_occ_calc,
        "unit_price": unit_price,
        "revenue_target": revenue_target,

        # 実績（あれば）
        "occ_actual": occ_actual,
        "revenue_actual": revenue_actual,
        "delta_revenue": delta_revenue,
        "patient_days_actual": patient_days_actual,
        "admissions_actual": admissions_actual,

        # 比較（あれば）
        "add_patient_days": add_patient_days,
        "add_admissions": add_admissions,
        "required_admissions": required_admissions,

        # 目標だけでも出せる
        "required_admissions_target": required_admissions_target,
    }


def read_monthly_table(uploaded_file):
    """期間集計用の月次テーブルを読み込む（CSV / Excel）"""
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.copy()

    # 旧名称→新名称（後方互換）
    if "入院収入（実績）" not in df.columns and "DPC入院収入（実績）" in df.columns:
        df = df.rename(columns={"DPC入院収入（実績）": "入院収入（実績）"})
    if "入院単価" not in df.columns and "日当" in df.columns:
        df = df.rename(columns={"日当": "入院単価"})

    required_cols = ["年月", "稼働病床数", "延べ患者数（人日）", "入院収入（実績）"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {missing}")

    df["年月"] = pd.to_datetime(df["年月"]).dt.to_period("M").dt.to_timestamp()
    df["月日数"] = df["年月"].apply(lambda x: int((x + pd.offsets.MonthEnd(0)).day))
    df["最大延べ患者数（100%）"] = pd.to_numeric(df["稼働病床数"], errors="coerce") * df["月日数"]

    for c in ["稼働病床数", "延べ患者数（人日）", "入院収入（実績）"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "入院単価" in df.columns:
        df["入院単価"] = pd.to_numeric(df["入院単価"], errors="coerce")
        mask = df["入院単価"].isna()
        df.loc[mask, "入院単価"] = df.loc[mask, "入院収入（実績）"] / df.loc[mask, "延べ患者数（人日）"]
    else:
        df["入院単価"] = df["入院収入（実績）"] / df["延べ患者数（人日）"]

    df = df.sort_values("年月").reset_index(drop=True)
    return df

# -------------------------
# UI
# -------------------------
st.title("KADO94")
st.caption("稼働率94%：入院収入シミュレーション & 固定費カバー率")

with st.sidebar:
    st.subheader("共通設定")
    input_mode = st.radio("シミュレーション", ["目標値だけ（実績不要）", "実績と比較（増収額まで）"], index=0)
    calc_mode = st.radio("計算モード", ["高精度", "Excel互換"], index=0, horizontal=True)
    graph_size = st.selectbox("グラフ表示サイズ", ["コンパクト", "標準", "ワイド（全幅）"], index=0)
    target_occ = st.slider("目標稼働率", min_value=0.50, max_value=1.00, value=0.94, step=0.01)
    st.caption("※ Excelの計算ロジックをPythonに移植して計算します（Excel計算エンジンは使いません）。")

tab_sim, tab_fc = st.tabs(["94%シミュレーション", "固定費カバー率・期間集計"])

# -------------------------
# 94% simulation (Main)
# -------------------------
with tab_sim:
    cs = _chart_settings(graph_size)
    st.subheader("月次シミュレーション")

    colL, colR = st.columns([1.1, 1.2], gap="large")
    with colL:
        month = st.date_input("年月（その月のどの日でもOK）", value=date.today().replace(day=1))
        month_start = to_month_start(month)
        beds = st.number_input("稼働病床数", min_value=0.0, value=401.0, step=1.0)
        los = st.number_input("平均在院日数", min_value=0.1, value=10.4, step=0.1)


        # --- 目標の設定（実績不要モードで使用） ---
        target_mode = "稼働率"
        target_admissions = None
        target_revenue = None

        st.caption(f"稼働病床数: {beds:,.0f} 床 / 平均在院日数: {los:.1f} 日")

        if input_mode == "目標値だけ（実績不要）":
            st.markdown("##### 目標の設定")
            target_mode = st.radio("目標の指定方法", ["稼働率", "新入院数", "入院収入"], horizontal=True)
            if target_mode == "新入院数":
                target_admissions = st.number_input("新入院数（目標）", min_value=0.0, value=900.0, step=1.0)
                st.caption(f"入力値: {target_admissions:,.0f} 人")
            elif target_mode == "入院収入":
                target_revenue = st.number_input("入院収入（目標・円）", min_value=0.0, value=1_000_000_000.0, step=1_000_000.0)
                st.caption(f"入力値: {yen(target_revenue)}")
            else:
                st.caption(f"入力値: 目標稼働率 {(target_occ*100):.1f}%")
        # --- 実績入力（任意） ---
        patient_days_actual = None
        admissions_actual = None
        revenue_actual = None

        if input_mode == "実績と比較（増収額まで）":
            st.markdown("##### 実績（比較する場合は入力）")
            patient_days_actual = st.number_input("延べ患者数（実績・人日）", min_value=0.0, value=10136.0, step=1.0)
            st.caption(f"入力値: {patient_days_actual:,.0f} 人日")
            admissions_actual = st.number_input("新入院数（実績）", min_value=0.0, value=916.0, step=1.0)
            st.caption(f"入力値: {admissions_actual:,.0f} 人")
            revenue_actual = st.number_input("入院収入（実績・円）", min_value=0.0, value=870_000_000.0, step=1_000_000.0)
            st.caption(f"入力値: {yen(revenue_actual)}")

        else:
            with st.expander("実績を入力して増収額も見たい（任意）"):
                patient_days_actual = st.number_input("延べ患者数（実績・人日）", min_value=0.0, value=0.0, step=1.0)
                st.caption(f"入力値: {patient_days_actual:,.0f} 人日")
                admissions_actual = st.number_input("新入院数（実績）", min_value=0.0, value=0.0, step=1.0)
                st.caption(f"入力値: {admissions_actual:,.0f} 人")
                revenue_actual = st.number_input("入院収入（実績・円）", min_value=0.0, value=0.0, step=1_000_000.0)
                st.caption(f"入力値: {yen(revenue_actual)}")

                # 未入力（0扱い）をNoneに変換：比較表示を抑制
                if patient_days_actual == 0:
                    patient_days_actual = None
                if admissions_actual == 0:
                    admissions_actual = None
                if revenue_actual == 0:
                    revenue_actual = None

    with colR:
        st.markdown("#### 入院単価")

        # 単価の決め方：実績があるときだけ自動計算が使える
        can_auto = (revenue_actual is not None) and (patient_days_actual is not None) and (patient_days_actual != 0)

        auto_unit = st.checkbox("入院単価（1人日あたり）を実績から自動計算する", value=can_auto, disabled=(not can_auto))

        if auto_unit and can_auto:
            unit_price_raw = (revenue_actual / patient_days_actual)
            if calc_mode == "Excel互換":
                unit_price = float(excel_round0(unit_price_raw))
                st.info(f"入院単価（自動・Excel互換）: {yen(unit_price)} / 人日（内部: {unit_price_raw:,.2f}）")
            else:
                unit_price = float(unit_price_raw)
                st.info(f"入院単価（自動）: {unit_price_raw:,.2f} 円/人日（表示: {yen(unit_price_raw)}）")
        else:
            unit_price = int_input_comma("入院単価（円/人日）", key="unit_price", default=85_911.0)
            st.caption(f"入力値: {yen(unit_price)} / 人日")

        if input_mode == "目標値だけ（実績不要）":
            st.caption("※「目標値だけ」では増収額などの比較指標は、実績を入力しない限り表示されません。")
    result = simulate_month(
            month_start=month_start,
            target_occ=target_occ,
            beds=beds,
            los=los,
            unit_price=unit_price,
            target_mode=target_mode,
            target_admissions=target_admissions,
            target_revenue=target_revenue,
            patient_days_actual=patient_days_actual,
            admissions_actual=admissions_actual,
            revenue_actual=revenue_actual,
        )

    st.divider()

    # --- 主要指標 ---
    occ_actual = result["occ_actual"]
    target_occ_calc = result["target_occ_calc"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("稼働率（実績）", f"{(occ_actual*100):.1f}%" if occ_actual is not None else "—")
    c2.metric("稼働率（目標）", f"{(target_occ_calc*100):.1f}%" if target_occ_calc is not None else "—")
    c3.metric("必要延べ患者数（目標）", f"{fmt_int_excel(result['required_patient_days_target'])} 人日")
    c4.metric("入院収入（目標）", yen(result["revenue_target"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("目標新入院数（推計）", f"{fmt_int_excel(result['required_admissions_target'])} 人" if result["required_admissions_target"] is not None else "—")
    c6.metric("増収額（目標−実績）", yen(result["delta_revenue"]) if result["delta_revenue"] is not None else "—")
    c7.metric("追加必要延べ患者数", f"{fmt_int_excel(result['add_patient_days'])} 人日" if result["add_patient_days"] is not None else "—")
    c8.metric("追加必要新入院（推計）", f"{fmt_int_excel(result['add_admissions'])} 人" if result["add_admissions"] is not None else "—")

    if result["required_admissions"] is not None:
        st.caption(f"必要新入院（実績加算）: {fmt_int_excel(result['required_admissions'])} 人")

    if input_mode == "目標値だけ（実績不要）" and result["delta_revenue"] is None:
        st.info("実績値を入れていないので、増収額や追加必要量は「—」表示です（必要なら上の『実績を入力して増収額も見たい』を開いて入力）。")

    st.markdown("#### グラフ（コンパクト）")

    # -------------------------
    # 稼働率（実績 or 目標）: 94%未満は赤
    # -------------------------
    occ_actual = result["occ_actual"]
    target_occ_calc = result["target_occ_calc"]
    occ_base = occ_actual if occ_actual is not None else (target_occ_calc if target_occ_calc is not None else 0.0)
    occ_title = "稼働率（実績）" if occ_actual is not None else "稼働率（目標）"
    occ_pct = float(occ_base) * 100.0
    occ_color = "#EF4444" if occ_pct < float(target_occ) * 100.0 else "#22C55E"

    fig_occ = go.Figure()
    fig_occ.add_trace(go.Indicator(
        mode="number+gauge",
        value=float(occ_pct),
        number={"suffix": "%", "font": {"size": 22}},
        title={"text": occ_title},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, 100]},
            "threshold": {"line": {"width": 4}, "value": float(target_occ * 100)},
            "bar": {"thickness": 0.14, "color": occ_color},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    cs = _chart_settings(graph_size)

    fig_occ.update_layout(template="plotly_dark", width=cs["w_main"] if not cs["use_container"] else None, height=cs["h_occ"], margin=dict(l=8, r=8, t=32, b=8))
    _plotly_center(fig_occ, cs["w_main"], cs["use_container"])

    # -------------------------
    # 入院収入：実績（任意） vs 目標
    # -------------------------
    labels = []
    values = []
    texts = []
    if revenue_actual is not None:
        labels.append("実績")
        values.append(float(revenue_actual))
        texts.append(yen(revenue_actual))

    labels.append("目標")
    values.append(float(result["revenue_target"]))
    texts.append(yen(result["revenue_target"]))

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        x=labels,
        y=values,
        text=texts,
        textposition="outside",
        cliponaxis=False,
        width=[0.35] * len(labels),
        marker_line_width=0,
    ))
    fig_rev.update_layout(template="plotly_dark", width=cs["w_main"] if not cs["use_container"] else None, height=cs["h_bar"], yaxis_title="入院収入（円）", margin=dict(l=8, r=8, t=12, b=8))
    _plotly_center(fig_rev, cs["w_main"], cs["use_container"])

    # -------------------------
    # 追加必要量（実績入力があるときだけ表示）
    # -------------------------
    if result["add_patient_days"] is not None and result["add_admissions"] is not None:
        fig_need = go.Figure()
        fig_need.add_trace(go.Bar(
            y=["追加必要延べ患者数（人日）", "追加必要新入院（推計）"],
            x=[float(result["add_patient_days"]), float(result["add_admissions"])],
            orientation="h",
            text=[f"{fmt_int_excel(result['add_patient_days'])}", f"{fmt_int_excel(result['add_admissions'])}"],
            textposition="outside",
            cliponaxis=False,
            width=0.42,
            marker_line_width=0,
        ))
        fig_need.update_layout(template="plotly_dark", width=cs["w_main"] if not cs["use_container"] else None, height=cs["h_need"], margin=dict(l=8, r=8, t=8, b=8))
        _plotly_center(fig_need, cs["w_main"], cs["use_container"])
    else:
        st.caption("※ 実績（延べ患者数・新入院数）を入力すると、追加必要量のグラフが表示されます。")

    st.markdown("#### 計算内訳")
    detail = pd.DataFrame(
        [
            ["月日数", result["month_days"], ""],
            ["延べ患者数（100%）", result["max_patient_days_100"], "稼働病床数 × 月日数"],
            ["必要延べ患者数（目標稼働率）", result["required_patient_days_target"], "延べ患者数（100%） × 目標稼働率"],
            ["入院単価（円/人日）", result["unit_price"], "実績収入 ÷ 実績延べ患者数（or 手入力）"],
            ["入院収入（目標稼働率）", result["revenue_target"], "必要延べ患者数 × 入院単価"],
            ["増収額（目標−実績）", result["delta_revenue"], "目標収入 − 実績収入"],
            ["追加必要延べ患者数", result["add_patient_days"], "max(0, 必要延べ患者数 − 実績延べ患者数)"],
            ["追加必要新入院（推計）", result["add_admissions"], "ceil(追加必要延べ患者数 ÷ 平均在院日数)"],
            ["必要新入院（推計）", result["required_admissions"], "実績新入院数 + 追加必要新入院"],
        ],
        columns=["項目", "値", "式/メモ"],
    )
    def fmt(v, key):
        if isinstance(v, (int, float)):
            if "収入" in key or "単価" in key:
                return yen(v)
            return f"{v:,.0f}"
        return v
    detail["値"] = [fmt(v, k) for v, k in zip(detail["値"], detail["項目"])]
    st.dataframe(detail, use_container_width=True, hide_index=True)

    # -------------------------
    # Fixed-cost coverage + Period
    # -------------------------
with tab_fc:
    st.subheader("固定費カバー率（単月）")

    colL, colR = st.columns([1.1, 1.2], gap="large")
    with colL:
        month = st.date_input("年月（その月のどの日でもOK）", value=date.today().replace(day=1), key="fc_month")
        month_start = to_month_start(month)
        beds = st.number_input("稼働病床数", min_value=0.0, value=401.0, step=1.0, key="fc_beds")
        patient_days_actual = st.number_input("延べ患者数（人日）", min_value=0.0, value=10297.0, step=1.0, key="fc_pd")
        admissions_actual = st.number_input("新入院数", min_value=0.0, value=985.0, step=1.0, key="fc_adm")
        los = st.number_input("平均在院日数", min_value=0.1, value=10.5, step=0.1, key="fc_los")

    with colR:
        st.markdown("#### コスト前提")
        revenue_actual = st.number_input("入院収入（実績・円）", min_value=0.0, value=891_168_000.0, step=1_000_000.0, key="fc_rev")
        auto_unit = st.checkbox("入院単価（1人日あたり）を実績から自動計算する", value=True, key="fc_auto_unit")
        if auto_unit:
            unit_price = (revenue_actual / patient_days_actual) if patient_days_actual else 0.0
            st.info(f"入院単価（自動）: {yen(unit_price)} / 人日")
        else:
            unit_price = int_input_comma("入院単価（円/人日）", key="unit_price", default=86_546.0)

        fixed_cost_month = st.number_input("固定費（月額・円）", min_value=0.0, value=0.0, step=1_000_000.0, key="fc_fixed")
        var_cost_rate = st.number_input("変動費率（0〜1）", min_value=0.0, max_value=1.0, value=0.325, step=0.005, key="fc_var")
        unit_price_scenario = st.number_input("入院単価シナリオ（円/人日）", min_value=0.0, value=90_000.0, step=100.0, key="fc_scn_unit")

        result = simulate_month(
            month_start=month_start,
            target_occ=target_occ,
            beds=beds,
            los=los,
            unit_price=unit_price,
            patient_days_actual=patient_days_actual,
            admissions_actual=admissions_actual,
            revenue_actual=revenue_actual,
        )

        # Fixed-cost coverage
        margin_actual = (revenue_actual * (1 - var_cost_rate)) if (revenue_actual is not None) else None
        coverage_actual = (margin_actual / fixed_cost_month) if (fixed_cost_month and margin_actual is not None) else None

        margin_target = result["revenue_target"] * (1 - var_cost_rate)
        coverage_target = (margin_target / fixed_cost_month) if fixed_cost_month else None

        revenue_scenario = result["required_patient_days_target"] * unit_price_scenario
        margin_scenario = revenue_scenario * (1 - var_cost_rate)
        coverage_scenario = (margin_scenario / fixed_cost_month) if fixed_cost_month else None

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("限界利益（実績）", yen(margin_actual))
    m2.metric("限界利益（稼働率目標）", yen(margin_target))
    m3.metric("限界利益（入院単価シナリオ）", yen(margin_scenario))

    k1, k2, k3 = st.columns(3)
    k1.metric("固定費カバー率（実績）", "" if coverage_actual is None else f"{coverage_actual:.2f} 倍")
    k2.metric("固定費カバー率（稼働率目標）", "" if coverage_target is None else f"{coverage_target:.2f} 倍")
    k3.metric("固定費カバー率（入院単価シナリオ）", "" if coverage_scenario is None else f"{coverage_scenario:.2f} 倍")

    st.divider()
    st.subheader("期間集計（半期・年度）")
    st.caption("月次データ（CSV/Excel）をアップロードして、期間の固定費カバー率や増収額を計算します。")

    up = st.file_uploader("月次データをアップロード（CSV / Excel）", type=["csv", "xlsx", "xls"], key="monthly_up")
    template_cols = ["年月", "稼働病床数", "延べ患者数（人日）", "入院収入（実績）", "入院単価"]
    st.download_button(
        "テンプレCSVをダウンロード（列だけ）",
        data=pd.DataFrame(columns=template_cols).to_csv(index=False).encode("utf-8-sig"),
        file_name="monthly_template.csv",
        mime="text/csv",
    )

    if up is not None:
        try:
            dfm = read_monthly_table(up)
            st.success(f"読み込みOK：{len(dfm)}行")
            st.dataframe(dfm, use_container_width=True, hide_index=True)

            min_m = dfm["年月"].min().to_pydatetime().date()
            max_m = dfm["年月"].max().to_pydatetime().date()

            p1, p2, p3 = st.columns([1, 1, 1.2])
            with p1:
                start_m = st.date_input("開始年月", value=min_m, min_value=min_m, max_value=max_m, key="start_m")
            with p2:
                end_m = st.date_input("終了年月", value=max_m, min_value=min_m, max_value=max_m, key="end_m")
            start_ts = to_month_start(start_m)
            end_ts = to_month_start(end_m)

            if start_ts > end_ts:
                st.error("開始年月が終了年月より後になっています。")
            else:
                dff = dfm[(dfm["年月"] >= start_ts) & (dfm["年月"] <= end_ts)].copy()
                months = dff["年月"].nunique()

                # --- グラフ用の派生列 ---
                dff["月"] = dff["年月"].apply(month_label)
                dff["実績稼働率"] = dff["延べ患者数（人日）"] / dff["最大延べ患者数（100%）"]
                dff["目標稼働率"] = target_occ

                # 目標稼働率時の入院収入（月次）
                dff["入院単価（月次）"] = dff["入院収入（実績）"] / dff["延べ患者数（人日）"]
                dff["入院収入（目標稼働率）"] = (dff["最大延べ患者数（100%）"] * target_occ) * dff["入院単価（月次）"]
                dff["増収額（目標−実績）"] = dff["入院収入（目標稼働率）"] - dff["入院収入（実績）"]

                cs2 = _chart_settings(graph_size)

                st.markdown("#### 月次推移（期間内）")

                dff = dff.sort_values("年月").copy()
                x_month = dff["月"].tolist()
                occ_values = (dff["実績稼働率"] * 100).tolist()
                occ_colors = ["#EF4444" if v < float(target_occ)*100 else "#22C55E" for v in occ_values]

                # 稼働率（実績 vs 目標）
                fig_occ_m = go.Figure()
                fig_occ_m.add_trace(go.Scatter(
                    x=x_month, y=occ_values,
                    mode="lines+markers",
                    name="稼働率（実績）",
                    line=dict(width=2),
                    marker=dict(size=6, color=occ_colors),
                    hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
                ))
                fig_occ_m.add_trace(go.Scatter(
                    x=x_month, y=(dff["目標稼働率"] * 100),
                    mode="lines",
                    name="稼働率（目標）",
                    line=dict(dash="dash", width=2),
                    hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
                ))
                cs2 = _chart_settings(graph_size)

                fig_occ_m.update_layout(template="plotly_dark", width=cs2["w_period"] if not cs2["use_container"] else None, height=cs2["h_line"], yaxis_title="稼働率（%）", margin=dict(l=8, r=8, t=10, b=8), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                _plotly_center(fig_occ_m, cs2["w_period"], cs2["use_container"])

                # 入院収入（実績 vs 目標）
                fig_rev_m = go.Figure()
                fig_rev_m.add_trace(go.Scatter(
                    x=x_month, y=dff["入院収入（実績）"],
                    mode="lines+markers",
                    name="入院収入（実績）",
                    line=dict(width=2),
                    marker=dict(size=6, color=occ_colors),
                    hovertemplate="%{x}<br>¥%{y:,.0f}<extra></extra>",
                ))
                fig_rev_m.add_trace(go.Scatter(
                    x=x_month, y=dff["入院収入（目標稼働率）"],
                    mode="lines+markers",
                    name="入院収入（目標）",
                    line=dict(dash="dash", width=2),
                    marker=dict(size=6, color=occ_colors),
                    hovertemplate="%{x}<br>¥%{y:,.0f}<extra></extra>",
                ))
                fig_rev_m.update_layout(template="plotly_dark", width=cs2["w_period"] if not cs2["use_container"] else None, height=cs2["h_line"] + 20, yaxis_title="入院収入（円）", margin=dict(l=8, r=8, t=10, b=8), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                _plotly_center(fig_rev_m, cs2["w_period"], cs2["use_container"])

                # 増収額（棒）
                fig_delta = go.Figure()
                fig_delta.add_trace(go.Bar(
                    x=x_month,
                    y=dff["増収額（目標−実績）"],
                    text=[f"¥{v:,.0f}" for v in dff["増収額（目標−実績）"]],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>¥%{y:,.0f}<extra></extra>",
                ))
                fig_delta.update_layout(template="plotly_dark", width=cs2["w_period"] if not cs2["use_container"] else None, height=cs2["h_line"], yaxis_title="増収額（円）", margin=dict(l=8, r=8, t=10, b=8))
                _plotly_center(fig_delta, cs2["w_period"], cs2["use_container"])

                # 固定費カバー率（月次）
                if fixed_cost_month and fixed_cost_month > 0:
                    fig_cov = go.Figure()
                    fig_cov.add_trace(go.Scatter(
                        x=x_month, y=dff["固定費カバー率（実績）"],
                        mode="lines+markers",
                        name="固定費カバー率（実績）",
                        line=dict(width=2),
                        marker=dict(size=6, color=occ_colors),
                        hovertemplate="%{x}<br>%{y:.2f}倍<extra></extra>",
                    ))
                    fig_cov.add_trace(go.Scatter(
                        x=x_month, y=dff["固定費カバー率（目標）"],
                        mode="lines+markers",
                        name="固定費カバー率（目標）",
                        line=dict(dash="dash", width=2),
                        marker=dict(size=6, color=occ_colors),
                        hovertemplate="%{x}<br>%{y:.2f}倍<extra></extra>",
                    ))
                    fig_cov.update_layout(template="plotly_dark", width=cs2["w_period"] if not cs2["use_container"] else None, height=cs2["h_line"], yaxis_title="固定費カバー率（倍）", margin=dict(l=8, r=8, t=10, b=8), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    _plotly_center(fig_cov, cs2["w_period"], cs2["use_container"])


                s1, s2, s3, s4 = st.columns(4)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("月数", f"{months} ヶ月")
                s2.metric("期間延べ患者数（実績）", f"{pd_actual:,.0f} 人日")
                s3.metric("期間入院収入（実績）", yen(rev_actual))
                s4.metric("期間入院単価（実績）", yen(unit_price_period))

                t1, t2, t3 = st.columns(3)
                t1.metric("期間必要延べ患者数（目標稼働率）", f"{req_pd:,.0f} 人日")
                t2.metric("期間入院収入（目標稼働率シミュ）", yen(rev_target))
                t3.metric("増収額（目標−実績）", yen(rev_target - rev_actual))

                u1, u2 = st.columns(2)
                u1.metric("固定費（期間）", yen(fixed_cost_period))
                u2.metric(
                    "固定費カバー率（期間・実績 / 目標）",
                    "" if cov_period_actual is None else f"{cov_period_actual:.2f} 倍 / {cov_period_target:.2f} 倍",
                )

                out = pd.DataFrame(
                    {
                        "開始年月": [start_ts.date()],
                        "終了年月": [end_ts.date()],
                        "月数": [months],
                        "期間延べ患者数（実績）": [pd_actual],
                        "期間入院収入（実績）": [rev_actual],
                        "期間入院単価（実績）": [unit_price_period],
                        "期間最大延べ患者数（100%）": [max_pd],
                        "期間必要延べ患者数（目標稼働率）": [req_pd],
                        "期間入院収入（目標稼働率）": [rev_target],
                        "増収額（目標−実績）": [rev_target - rev_actual],
                        "固定費（月額）": [fixed_cost_month],
                        "固定費（期間）": [fixed_cost_period],
                        "変動費率": [var_cost_rate],
                        "限界利益（期間・実績）": [margin_period_actual],
                        "限界利益（期間・目標）": [margin_period_target],
                        "固定費カバー率（期間・実績）": [cov_period_actual],
                        "固定費カバー率（期間・目標）": [cov_period_target],
                    }
                )
                csv = out.to_csv(index=False).encode("utf-8-sig")
                st.download_button("期間集計結果をCSVでダウンロード", data=csv, file_name="period_summary.csv", mime="text/csv")

        except Exception as e:
            st.error(f"読み込みエラー: {e}")
