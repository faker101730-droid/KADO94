import math
from datetime import date
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
st.set_page_config(page_title="KADO94", layout="wide")

# -------------------------
# Helpers
# -------------------------
def to_month_start(d: date) -> pd.Timestamp:
    ts = pd.to_datetime(d)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def days_in_month(month_start: pd.Timestamp) -> int:
    return int((month_start + pd.offsets.MonthEnd(0)).day)

def yen(x):
    if x is None or (isinstance(x, float) and (pd.isna(x))):
        return ""
    try:
        return f"¥{int(round(float(x))):,}"
    except Exception:
        return str(x)



def month_label(ts: pd.Timestamp) -> str:
    try:
        return ts.strftime('%Y-%m')
    except Exception:
        return str(ts)


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
    patient_days_actual: float,
    admissions_actual: float,
    los: float,
    revenue_actual: float,
    unit_price: float,  # 入院単価（円/人日）
):
    d = days_in_month(month_start)
    max_patient_days = beds * d
    required_patient_days = max_patient_days * target_occ

    occ_actual = (patient_days_actual / max_patient_days) if max_patient_days else 0.0
    revenue_target = required_patient_days * unit_price
    delta_revenue = revenue_target - revenue_actual

    add_patient_days = max(0.0, required_patient_days - patient_days_actual)
    add_admissions = math.ceil(add_patient_days / los) if los and los > 0 else 0
    required_admissions = admissions_actual + add_admissions

    return {
        "month_start": month_start,
        "month_days": d,
        "max_patient_days_100": max_patient_days,
        "required_patient_days_target": required_patient_days,
        "occ_actual": occ_actual,
        "revenue_target": revenue_target,
        "delta_revenue": delta_revenue,
        "add_patient_days": add_patient_days,
        "add_admissions": add_admissions,
        "required_admissions": required_admissions,
        "unit_price": unit_price,
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
    calc_mode = st.radio("計算モード", ["高精度", "Excel互換"], index=0, horizontal=True)
    target_occ = st.slider("目標稼働率", min_value=0.50, max_value=1.00, value=0.94, step=0.01)
    st.caption("※ Excelの計算ロジックをPythonに移植して計算します（Excel計算エンジンは使いません）。")

tab_sim, tab_fc = st.tabs(["94%シミュレーション", "固定費カバー率・期間集計"])

# -------------------------
# 94% simulation (Main)
# -------------------------
with tab_sim:
    st.subheader("月次シミュレーション")

    colL, colR = st.columns([1.1, 1.2], gap="large")
    with colL:
        month = st.date_input("年月（その月のどの日でもOK）", value=date.today().replace(day=1))
        month_start = to_month_start(month)
        beds = st.number_input("稼働病床数", min_value=0.0, value=401.0, step=1.0)
        patient_days_actual = st.number_input("延べ患者数（人日）", min_value=0.0, value=10136.0, step=1.0)
        admissions_actual = st.number_input("新入院数", min_value=0.0, value=916.0, step=1.0)
        los = st.number_input("平均在院日数", min_value=0.1, value=10.4, step=0.1)

    with colR:
        st.markdown("#### 実績収入と入院単価")
        revenue_actual = st.number_input("入院収入（実績・円）", min_value=0.0, value=870_000_000.0, step=1_000_000.0)
        auto_unit = st.checkbox("入院単価（1人日あたり）を実績から自動計算する", value=True)
        if auto_unit:
            unit_price = (revenue_actual / patient_days_actual) if patient_days_actual else 0.0
            st.info(f"入院単価（自動）: {yen(unit_price)} / 人日")
        else:
            unit_price = st.number_input("入院単価（円/人日）", min_value=0.0, value=85_911.0, step=100.0)

        result = simulate_month(
            month_start=month_start,
            target_occ=target_occ,
            beds=beds,
            patient_days_actual=patient_days_actual,
            admissions_actual=admissions_actual,
            los=los,
            revenue_actual=revenue_actual,
            unit_price=unit_price,
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("実績稼働率", f"{result['occ_actual']*100:.1f}%")
    c2.metric("必要延べ患者数（目標稼働率）", f"{result['required_patient_days_target']:,.0f} 人日")
    c3.metric("入院収入（目標稼働率）", yen(result["revenue_target"]))
    c4.metric("増収額（目標−実績）", yen(result["delta_revenue"]))

    c5, c6, c7 = st.columns(3)
    c5.metric("追加必要延べ患者数", f"{result['add_patient_days']:,.0f} 人日")
    c6.metric("追加必要新入院（推計）", f"{result['add_admissions']:,.0f} 人")
    c7.metric("必要新入院（推計）", f"{result['required_admissions']:,.0f} 人")


    st.markdown("#### グラフ（実績 vs 目標）")

    # 稼働率：バレット（実績バー）＋目標ライン
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Indicator(
        mode="number+gauge",
        value=float(result["occ_actual"] * 100),
        number={"suffix": "%", "font": {"size": 26}},
        title={"text": "稼働率（実績）"},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, 100]},
            "threshold": {"line": {"width": 4}, "value": float(target_occ * 100)},
            "bar": {"thickness": 0.18},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_occ.update_layout(template="plotly_dark", height=110, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig_occ, use_container_width=True)

    # 入院収入：実績 vs 目標（数値ラベル）
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        x=["実績", "目標"],
        y=[float(revenue_actual), float(result["revenue_target"])],
        text=[yen(revenue_actual), yen(result["revenue_target"])],
        textposition="outside",
        cliponaxis=False,
        width=[0.35, 0.35],
    ))
    fig_rev.update_layout(
        template="plotly_dark",
        height=260,
        yaxis_title="入院収入（円）",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # 追加必要量：横棒（ラベル付き）
    fig_need = go.Figure()
    fig_need.add_trace(go.Bar(
        y=["追加必要延べ患者数（人日）", "追加必要新入院（推計）"],
        x=[float(result["add_patient_days"]), float(result["add_admissions"])],
        orientation="h",
        text=[f"{result['add_patient_days']:,.0f}", f"{result['add_admissions']:,.0f}"],
        textposition="outside",
        cliponaxis=False,
    ))
    fig_need.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_need, use_container_width=True)

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
            unit_price = st.number_input("入院単価（円/人日）", min_value=0.0, value=86_546.0, step=100.0, key="fc_unit_price")

        fixed_cost_month = st.number_input("固定費（月額・円）", min_value=0.0, value=0.0, step=1_000_000.0, key="fc_fixed")
        var_cost_rate = st.number_input("変動費率（0〜1）", min_value=0.0, max_value=1.0, value=0.325, step=0.005, key="fc_var")
        unit_price_scenario = st.number_input("入院単価シナリオ（円/人日）", min_value=0.0, value=90_000.0, step=100.0, key="fc_scn_unit")

        result = simulate_month(
            month_start=month_start,
            target_occ=target_occ,
            beds=beds,
            patient_days_actual=patient_days_actual,
            admissions_actual=admissions_actual,
            los=los,
            revenue_actual=revenue_actual,
            unit_price=unit_price,
        )

        # Fixed-cost coverage
        margin_actual = revenue_actual * (1 - var_cost_rate)
        coverage_actual = (margin_actual / fixed_cost_month) if fixed_cost_month else None

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

                st.markdown("#### 月次推移（期間内）")

                dff = dff.sort_values("年月").copy()
                x_month = dff["月"].tolist()

                # 稼働率（実績 vs 目標）
                fig_occ_m = go.Figure()
                fig_occ_m.add_trace(go.Scatter(
                    x=x_month, y=(dff["実績稼働率"] * 100),
                    mode="lines+markers",
                    name="稼働率（実績）",
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
                ))
                fig_occ_m.add_trace(go.Scatter(
                    x=x_month, y=(dff["目標稼働率"] * 100),
                    mode="lines",
                    name="稼働率（目標）",
                    line=dict(dash="dash", width=2),
                    hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
                ))
                fig_occ_m.update_layout(
                    template="plotly_dark",
                    height=280,
                    yaxis_title="稼働率（%）",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_occ_m, use_container_width=True)

                # 入院収入（実績 vs 目標）
                fig_rev_m = go.Figure()
                fig_rev_m.add_trace(go.Scatter(
                    x=x_month, y=dff["入院収入（実績）"],
                    mode="lines+markers",
                    name="入院収入（実績）",
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>¥%{y:,.0f}<extra></extra>",
                ))
                fig_rev_m.add_trace(go.Scatter(
                    x=x_month, y=dff["入院収入（目標稼働率）"],
                    mode="lines+markers",
                    name="入院収入（目標）",
                    line=dict(dash="dash", width=2),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>¥%{y:,.0f}<extra></extra>",
                ))
                fig_rev_m.update_layout(
                    template="plotly_dark",
                    height=300,
                    yaxis_title="入院収入（円）",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_rev_m, use_container_width=True)

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
                fig_delta.update_layout(
                    template="plotly_dark",
                    height=280,
                    yaxis_title="増収額（円）",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_delta, use_container_width=True)

                # 固定費カバー率（月次）
                if fixed_cost_month and fixed_cost_month > 0:
                    fig_cov = go.Figure()
                    fig_cov.add_trace(go.Scatter(
                        x=x_month, y=dff["固定費カバー率（実績）"],
                        mode="lines+markers",
                        name="固定費カバー率（実績）",
                        line=dict(width=2),
                        marker=dict(size=6),
                        hovertemplate="%{x}<br>%{y:.2f}倍<extra></extra>",
                    ))
                    fig_cov.add_trace(go.Scatter(
                        x=x_month, y=dff["固定費カバー率（目標）"],
                        mode="lines+markers",
                        name="固定費カバー率（目標）",
                        line=dict(dash="dash", width=2),
                        marker=dict(size=6),
                        hovertemplate="%{x}<br>%{y:.2f}倍<extra></extra>",
                    ))
                    fig_cov.update_layout(
                        template="plotly_dark",
                        height=280,
                        yaxis_title="固定費カバー率（倍）",
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_cov, use_container_width=True)


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
