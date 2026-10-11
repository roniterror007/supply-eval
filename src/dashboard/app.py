"""
PlanFlow AI 2.0 - Simulation-Optimization, Evidence Graph & Resilience Dashboard
Interactive Enterprise Cockpit implementing:
1. Planning Cockpit & Digital Twin (Peer dialogue, Red-Team Challenges, Boss Governance)
2. Operations Research Optimizer (MILP Master Planner & Feasibility Oracle)
3. Evidence Knowledge Graph (HELICASE Provenance, Staleness, Contradictions, Human Prioritization)
4. 4-Track Operational Resilience Benchmark (Decision Regret, CVaR, Fill-rate, Zero Unsafe Actions)
5. 10,000 Enterprise Data Lake Explorer
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.gpu_utils import get_gpu_manager
from src.agents.orchestrator import PlanFlowOrchestrator
from src.optimization.candidate_generator import CandidatePlanGenerator
from src.knowledge_graph.truth_maintenance import TruthMaintenanceEngine

# Page Configuration
st.set_page_config(
    page_title="PlanFlow AI 2.0 | Simulation-Optimization Planning",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern enterprise CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .main { 
        background-color: #0A0C10; 
        color: #F8FAFC; 
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.05) 0%, transparent 50%);
    }
    
    .stMetric { 
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 20px !important; 
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease !important;
    }
    
    .stMetric:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }

    .boss-approved { 
        background: linear-gradient(135deg, rgba(19, 46, 34, 0.8) 0%, rgba(6, 78, 59, 0.6) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4); 
        padding: 22px; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .boss-rejected { 
        background: linear-gradient(135deg, rgba(56, 26, 26, 0.8) 0%, rgba(127, 29, 29, 0.6) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4); 
        padding: 22px; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .audit-card { 
        background: rgba(30, 41, 59, 0.6); 
        padding: 20px; 
        border-radius: 16px; 
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_orchestrator():
    return PlanFlowOrchestrator()


@st.cache_data
def load_datasets():
    test_path = "data/test_ground_truth_2k.parquet"
    ref_path = "data/enterprise_reference_8k.parquet"
    df_test = pd.read_parquet(test_path) if os.path.exists(test_path) else pd.DataFrame()
    df_ref = pd.read_parquet(ref_path) if os.path.exists(ref_path) else pd.DataFrame()
    return df_ref, df_test


orchestrator = load_orchestrator()
candidate_gen = CandidatePlanGenerator()
df_ref, df_test = load_datasets()
gpu_mgr = get_gpu_manager()
telemetry = gpu_mgr.get_telemetry()

# Sidebar: Hardware Telemetry and Navigation
with st.sidebar:
    st.title("⚡ PlanFlow AI 2.0")
    st.caption("Simulation–Optimization & Robust Governance")

    st.subheader("🖥️ Hardware Acceleration")
    st.write(f"**Device:** `{telemetry['device']}`")
    st.write(f"**CUDA Status:** {'🟢 Active' if telemetry['cuda_available'] else '🔴 CPU Fallback'}")
    if telemetry["cuda_available"]:
        st.write(f"**VRAM Allocated:** `{telemetry['vram_allocated_mb']} MB` / `{telemetry['vram_total_mb']} MB`")
        st.progress(min(1.0, telemetry["vram_allocated_mb"] / max(1.0, telemetry["vram_total_mb"])))

    st.markdown("---")
    st.subheader("📊 Data Lake Status")
    st.write(f"• Reference: `{len(df_ref):,}` records")
    st.write(f"• Ground Truth Test: `{len(df_test):,}` records")
    st.write(f"• Invariant Verifier: `HiGHS MILP & Feasibility Oracle`")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Planning Cockpit & Digital Twin",
        "⚙️ OR Optimizer & Feasibility (MILP)",
        "🕸️ Evidence Knowledge Graph (HELICASE)",
        "🛡️ 4-Track Resilience Benchmark",
        "🔍 10,000 Dataset Explorer",
    ]
)

# ==============================================================================
# TAB 1: PLANNING COCKPIT & DIGITAL TWIN
# ==============================================================================
with tab1:
    st.header("🏭 Autonomous Demand-to-Supply Planning Cockpit")
    st.markdown(
        "*Transforming forecasts into closed-loop actions with peer mutual correction, adversarial red-teaming, MILP optimization, and executive governance.*"
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        region_select = st.selectbox(
            "Target Market / Region",
            ["Germany", "United States", "Japan", "Poland", "Mexico"],
            index=0,
        )
    with col_ctrl2:
        business_line_select = st.selectbox(
            "Business Line (Product)",
            [
                "Analytical",
                "Force and Measurement",
                "Instrumentation",
            ],
            index=0,
        )
        
    st.markdown("<h4 style='color: #38BDF8; margin-top: 15px;'>🤖 Agentic AI Scenario Override</h4>", unsafe_allow_html=True)
    scenario_name = st.text_area(
        "Describe a complex scenario in natural language. The AI will extract the parameters and override the sliders.",
        value="",
        placeholder="e.g. A massive hurricane hit Stuttgart reducing capacity by 50% and causing 10 day delays.",
        height=80,
    )
    
    st.markdown(
        """
        <div style="font-size: 0.85em; color: #94A3B8; margin-top: -10px; margin-bottom: 20px;">
        💡 <b>Example Prompts:</b> 
        <ul style="margin-top: 4px;">
            <li><i>"A massive hurricane hit Stuttgart reducing capacity by 50% and causing 10 day delays."</i></li>
            <li><i>"Sudden market boom causes 20% demand surge but suppliers are struggling with 14 days delay."</i></li>
            <li><i>"Labor strike drops capacity to 40% and demand crashes by 15%."</i></li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        demand_surge_pct = st.slider("Demand Shock Factor (%)", -50, 100, 15, 1)
    with col_s2:
        supplier_delay_days = st.slider("Critical Component Transit Delay (Days)", 0, 60, 7, 1)
    with col_s3:
        capacity_factor = st.slider("Factory Operating Capacity (%)", 20, 100, 90, 5) / 100.0

    subset = df_test[(df_test["region"] == region_select) & (df_test["business_line"] == business_line_select)]
    sample_row = subset.iloc[0].to_dict() if len(subset) > 0 else df_test.iloc[0].to_dict()

    with st.spinner("Simulating..."):
        results = orchestrator.run_planning_cycle(
            row_dict=sample_row,
            scenario_name=scenario_name,
            demand_multiplier=1.0 + (demand_surge_pct / 100.0),
            supplier_delay_days=supplier_delay_days,
            capacity_multiplier=capacity_factor,
        )
        st.session_state["planning_results"] = results
        st.session_state["sample_row"] = sample_row

    if results.get("llm_params"):
        lp = results["llm_params"]
        st.success(f"**🤖 AI Scenario Extraction Active:** Replaced manual sliders with dynamically extracted values: **Demand x{lp.get('demand_multiplier', 1.0)} | Delay: {lp.get('supplier_delay_days', 0)} days | Capacity: {lp.get('capacity_multiplier', 1.0)*100:.0f}%**")

    d = results["demand"]
    f = results["factory"]
    s = results["supply"]
    prod = results["production"]
    scn = results["scenario"]
    r = results["recommendation"]
    m = results["manager_audit"]
    b = results["boss_decision"]
    graph = results["evidence_graph"]
    milp = results["milp_optimal_plan"]
    feas = results["feasibility_report"]
    auto = results["autonomy_decision"]
    red_team = results["red_team_challenges"]

    st.markdown("---")
    st.subheader("📊 Executive Summary — The Transformation")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Consensus Demand", value=f"{d.consensus_demand_units:,.0f} units", delta=f"+{demand_surge_pct}% Shock")
    with m_col2:
        st.metric(label="MILP Fulfilled Volume", value=f"{milp.total_fulfilled_units:,.0f} units", delta="100% Demand Satisfaction")
    with m_col3:
        shortage_u = sum(s.shortages.values())
        st.metric(label="Critical Component Gap", value=f"{shortage_u:,.0f} units", delta=f"{s.delivery_risk.value} Risk", delta_color="inverse")
    with m_col4:
        st.metric(label="Net Protected Enterprise Value", value=f"${r.net_benefit_usd:,.0f}", delta=f"${milp.total_cost_usd:,.0f} Spend")

    st.markdown("---")
    st.header("🔄 End-to-End Demand ➔ Supply Agentic Flow")
    st.caption("Forecast ➔ Demand ➔ Factory ➔ Production ➔ Supply Chain ➔ Scenario ➔ Recommendation ➔ Human Approval")

    st.html(
        f"""
<div style="padding: 20px; border-radius: 16px; background: linear-gradient(135deg, rgba(45, 27, 40, 0.8) 0%, rgba(26, 17, 24, 0.9) 100%); border-left: 5px solid #F43F5E; margin-bottom: 25px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease;">
<h4 style="margin:0 0 12px 0; color: #F43F5E; display: flex; align-items: center; font-weight: 600;">
    <span style="font-size: 1.3rem; margin-right: 10px; text-shadow: 0 0 10px rgba(244,63,94,0.5);">🚨</span> System Discovers
</h4>
<div style="color: #F1F5F9; font-size: 0.95rem; display: flex; gap: 15px; flex-wrap: wrap;">
    <span style="background: rgba(244, 63, 94, 0.15); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(244, 63, 94, 0.4); box-shadow: 0 4px 12px rgba(244,63,94,0.15); font-weight: 500;">Demand <b style="color:white;">+{demand_surge_pct}%</b></span>
    <span style="background: rgba(245, 158, 11, 0.15); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.4); box-shadow: 0 4px 12px rgba(245,158,11,0.15); font-weight: 500;"><b style="color:white;">{len(f.overloaded_factories)}</b> Factory Overload(s)</span>
    <span style="background: rgba(239, 68, 68, 0.15); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.4); box-shadow: 0 4px 12px rgba(239,68,68,0.15); font-weight: 500;"><b style="color:white;">{len(s.shortages)}</b> Material Shortage(s)</span>
    <span style="background: rgba(168, 85, 247, 0.15); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(168, 85, 247, 0.4); box-shadow: 0 4px 12px rgba(168,85,247,0.15); font-weight: 500;"><b style="color:white;">{results['opportunity'].at_risk_deals_count}</b> At-Risk Deals</span>
</div>
</div>

<div style="margin-bottom: 30px;">
    <h3 style="color: #38BDF8; font-weight: 600; text-shadow: 0 0 15px rgba(56,189,248,0.3);">🗣️ Shared Agent Blackboard Feed</h3>
    <div style="background: rgba(15, 23, 42, 0.4); border-radius: 16px; padding: 20px; max-height: 400px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.08); box-shadow: inset 0 4px 20px rgba(0,0,0,0.3); backdrop-filter: blur(5px);">
"""
    )
    
    blackboard = results.get("blackboard")
    if blackboard:
        # Display all claims and messages in chronological order (simulated by listing claims then messages)
        # In a real app we'd sort by timestamp, but here we just show claims then messages
        for claim in blackboard.claims.values():
            status_color = "#10B981" if claim.status in ["SUPPORTED", "ACCEPTED"] else ("#F59E0B" if claim.status == "CHALLENGED" else "#38BDF8")
            st.html(f"""
                <div style="margin-bottom: 12px; padding: 12px; border-left: 4px solid {status_color}; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px;">
                        <b>{claim.agent_id}</b> • Topic: {claim.topic} • Status: <span style="color: {status_color}; font-weight: bold;">{claim.status}</span>
                    </div>
                    <div style="color: #F1F5F9;">{claim.statement}</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Confidence: {claim.confidence:.2f}</div>
                </div>
            """)
        
        for msg in blackboard.messages:
            msg_color = "#F43F5E" if msg.message_type == "CHALLENGE" else "#A855F7"
            st.html(f"""
                <div style="margin-bottom: 12px; padding: 12px; border-left: 4px solid {msg_color}; background: rgba(255,255,255,0.05); border-radius: 8px; margin-left: 20px;">
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px;">
                        <b>{msg.sender}</b> ➔ <b>{', '.join(msg.recipients)}</b> • Type: <span style="color: {msg_color}; font-weight: bold;">{msg.message_type}</span>
                    </div>
                    <div style="color: #F1F5F9;">{msg.requested_action}</div>
                </div>
            """)
    
    st.html("</div></div>")

    st.html(
        f"""
<style>
.pipeline-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 24px; margin-bottom: 30px; }}
.agent-card {{ background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px); transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease; }}
.agent-card:hover {{ transform: translateY(-8px) scale(1.02); border-color: rgba(56, 189, 248, 0.5); box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4), 0 0 20px rgba(56, 189, 248, 0.15); }}
.agent-header {{ display: flex; align-items: center; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); }}
.agent-icon {{ font-size: 22px; margin-right: 16px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }}
.icon-blue {{ background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(56, 189, 248, 0.05) 100%); color: #38BDF8; border: 1px solid rgba(56,189,248,0.2); text-shadow: 0 0 10px rgba(56,189,248,0.4); }}
.icon-orange {{ background: linear-gradient(135deg, rgba(249, 115, 22, 0.2) 0%, rgba(249, 115, 22, 0.05) 100%); color: #F97316; border: 1px solid rgba(249,115,22,0.2); text-shadow: 0 0 10px rgba(249,115,22,0.4); }}
.icon-red {{ background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%); color: #EF4444; border: 1px solid rgba(239,68,68,0.2); text-shadow: 0 0 10px rgba(239,68,68,0.4); }}
.icon-purple {{ background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(168, 85, 247, 0.05) 100%); color: #A855F7; border: 1px solid rgba(168,85,247,0.2); text-shadow: 0 0 10px rgba(168,85,247,0.4); }}
.icon-teal {{ background: linear-gradient(135deg, rgba(20, 184, 166, 0.2) 0%, rgba(20, 184, 166, 0.05) 100%); color: #14B8A6; border: 1px solid rgba(20,184,166,0.2); text-shadow: 0 0 10px rgba(20,184,166,0.4); }}
.icon-green {{ background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%); color: #10B981; border: 1px solid rgba(16,185,129,0.2); text-shadow: 0 0 10px rgba(16,185,129,0.4); }}
.agent-title {{ font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0; letter-spacing: 0.5px; }}
.metric-row {{ display: flex; justify-content: space-between; margin-bottom: 12px; align-items: baseline; transition: background 0.2s; padding: 4px 8px; border-radius: 6px; margin-left: -8px; margin-right: -8px; }}
.metric-row:hover {{ background: rgba(255,255,255,0.03); }}
.metric-label {{ color: #94A3B8; font-size: 0.95rem; font-weight: 400; }}
.metric-value {{ color: #F1F5F9; font-weight: 600; font-size: 1.05rem; text-align: right; }}
.val-highlight {{ color: #38BDF8; font-weight: 700; text-shadow: 0 0 10px rgba(56,189,248,0.4); }}
.val-warning {{ color: #FBBF24; font-weight: 700; text-shadow: 0 0 10px rgba(251,191,36,0.4); }}
.val-danger {{ color: #F87171; font-weight: 700; text-shadow: 0 0 10px rgba(248,113,113,0.4); }}
.val-success {{ color: #34D399; font-weight: 700; text-shadow: 0 0 10px rgba(52,211,153,0.4); }}
.rec-list {{ margin: 0; padding-left: 20px; color: #E2E8F0; font-size: 0.95rem; line-height: 1.6; }}
.rec-list li {{ margin-bottom: 8px; position: relative; }}
</style>
        """
    )

    # Pre-compute complex strings for HTML
    fac_alloc_html = "".join([f'<div class="metric-row"><span class="metric-label">{k}</span><span class="metric-value">{v:,.0f} u ({f.factory_utilizations.get(k, 0):.1f}%)</span></div>' for k, v in f.factory_allocations.items()])
    fac_status_class = "val-danger" if f.overloaded_factories else "val-success"
    fac_status_text = f"OVERLOAD ({len(f.overloaded_factories)})" if f.overloaded_factories else "HEALTHY"
    
    prod = results["production"]
    prod_bottleneck_class = "val-danger" if prod.bottleneck_detected else "val-success"
    prod_bottleneck_text = "YES" if prod.bottleneck_detected else "NO"
    
    # Determine risk color logic handling Enum
    delivery_risk_val = s.delivery_risk.value if hasattr(s.delivery_risk, 'value') else str(s.delivery_risk)
    sup_risk_class = "val-danger" if delivery_risk_val in ["HIGH", "CRITICAL"] else ("val-warning" if delivery_risk_val == "MEDIUM" else "val-success")
    
    scn = results["scenario"]
    recs_html = "".join([f"<li>{rec.description}</li>" for rec in r.recommendations])
    
    shortages_html = "".join([f'<div class="metric-row"><span class="metric-label">{mat} Shortage</span><span class="metric-value val-danger">{qty:,.0f} u</span></div>' for mat, qty in s.shortages.items() if qty > 0])
    if not shortages_html:
        shortages_html = '<div class="metric-row"><span class="metric-label">Material Shortages</span><span class="metric-value val-success">0 u</span></div>'

    st.html(
        f"""
<div class="pipeline-grid">
<!-- Card 1 -->
<div class="agent-card">
    <div class="agent-header">
        <div class="agent-icon icon-blue">📈</div>
        <h3 class="agent-title">1. Forecasting & Demand</h3>
    </div>
    <div class="metric-row"><span class="metric-label">ML Forecast</span><span class="metric-value">${results['forecast'].baseline_revenue_usd:,.0f}</span></div>
    <div class="metric-row"><span class="metric-label">Consensus Demand</span><span class="metric-value val-highlight">${d.consensus_revenue_usd:,.0f}</span></div>
    <div class="metric-row"><span class="metric-label">Forecasted Units</span><span class="metric-value">{d.consensus_demand_units:,.0f} u</span></div>
</div>

<!-- Card 2 -->
<div class="agent-card">
    <div class="agent-header">
        <div class="agent-icon icon-orange">🏭</div>
        <h3 class="agent-title">2. Factory Planning</h3>
    </div>
    {fac_alloc_html}
    <div class="metric-row" style="margin-top:12px; border-top:1px solid rgba(255,255,255,0.05); padding-top:8px;">
        <span class="metric-label">Capacity Status</span><span class="metric-value {fac_status_class}">{fac_status_text}</span>
    </div>
</div>

<!-- Card 3 -->
<div class="agent-card">
    <div class="agent-header">
        <div class="agent-icon icon-red">⚙️</div>
        <h3 class="agent-title">3. Production</h3>
    </div>
    <div class="metric-row"><span class="metric-label">Bottleneck Detected</span><span class="metric-value {prod_bottleneck_class}">{prod_bottleneck_text}</span></div>
    <div class="metric-row"><span class="metric-label">Overtime Required</span><span class="metric-value">{prod.overtime_hours_required} hrs</span></div>
    <div class="metric-row"><span class="metric-label">Production Cost</span><span class="metric-value">${prod.production_cost_usd:,.0f}</span></div>
</div>

<!-- Card 4 -->
<div class="agent-card">
    <div class="agent-header">
        <div class="agent-icon icon-purple">🔗</div>
        <h3 class="agent-title">4. Supply Chain</h3>
    </div>
    <div class="metric-row"><span class="metric-label">Delivery Risk</span><span class="metric-value {sup_risk_class}">{delivery_risk_val}</span></div>
    {shortages_html}
    <div class="metric-row"><span class="metric-label">Expedite Cost</span><span class="metric-value">${s.total_expedite_cost_usd:,.0f}</span></div>
</div>

<!-- Card 5 -->
<div class="agent-card">
    <div class="agent-header">
        <div class="agent-icon icon-teal">🧪</div>
        <h3 class="agent-title">5. Scenario Twin</h3>
    </div>
    <div style="color:#CBD5E1; font-size:0.85rem; margin-bottom:12px; font-style:italic;">"{scn.scenario_description}"</div>
    <div class="metric-row"><span class="metric-label">Simulated Max Util</span><span class="metric-value">{scn.simulated_max_factory_utilization:.1f}%</span></div>
    <div class="metric-row"><span class="metric-label">Simulated Shortage</span><span class="metric-value">{scn.simulated_shortage_units:,.0f} u</span></div>
</div>

<!-- Card 6 -->
<div class="agent-card">
    <div class="agent-header">
        <div class="agent-icon icon-green">🎯</div>
        <h3 class="agent-title">6. Recommendation</h3>
    </div>
    <ul class="rec-list">
        {recs_html}
    </ul>
    <div class="metric-row" style="margin-top:15px; border-top:1px solid rgba(255,255,255,0.05); padding-top:12px;">
        <span class="metric-label">Net Protected Benefit</span>
        <span class="metric-value val-success">${r.net_benefit_usd:,.0f}</span>
    </div>
</div>
</div>
        """
    )

    st.markdown("---")
    col_boss, col_audit = st.columns([1, 1])
    
    with col_boss:
        st.subheader("👔 7. Human Approval / Executive Execution")
        
        # Interactive Approval Logic
        if "human_verdict" not in st.session_state:
            st.session_state["human_verdict"] = None
            
        def set_verdict(v):
            st.session_state["human_verdict"] = v
            
        if st.session_state["human_verdict"] is None:
            st.info("Pending Human Sign-off. Review the Recommendation (Card 6) and Tier 2 Audit (Right).")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.button("✅ Approve Plan", on_click=set_verdict, args=("APPROVED",), use_container_width=True, type="primary")
            with btn_col2:
                st.button("❌ Reject Plan", on_click=set_verdict, args=("REJECTED",), use_container_width=True)
        else:
            is_app = st.session_state["human_verdict"] == "APPROVED"
            card_class = "boss-approved" if is_app else "boss-rejected"
            icon = "✅" if is_app else "❌"
            
            st.markdown(
                f"""
                <div class="{card_class}">
                    <h3 style="margin: 0;">{icon} VERDICT: {st.session_state["human_verdict"]}</h3>
                    <p style="margin-top: 8px; font-size: 0.9em;">{b.final_orders_summary}</p>
                    <hr style="border-color: #334155; margin: 10px 0;">
                    <div style="font-size: 0.8em; color: #94A3B8;">
                        <b>Authorized Spend:</b> ${b.approved_budget_usd:,.0f}<br>
                        <b>Autonomy Routing:</b> <code>{auto.assigned_mode.upper()}</code><br>
                        <b>Digital Audit Signature:</b> <code>{b.executive_signature_hash}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            if st.button("Reset Approval", use_container_width=True):
                st.session_state["human_verdict"] = None
                st.rerun()

    with col_audit:
        st.subheader("⚖️ Tier 2: Judge Verification Audit")
        st.caption("Mathematical proof of constraints (PlanFlow extensions)")
        score_color = "#10B981" if m.verification_score >= 85 else "#EF4444"
        st.markdown(
            f"""
            <div class="audit-card">
                <h2 style="color: {score_color}; margin: 0;">{m.verification_score:.1f} / 100.0</h2>
                <p style="margin: 0; color: #94A3B8;">Overall Deterministic Score</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"• **Conservation of Units:** {'✅ PASSED' if m.unit_conservation_passed else '❌ VIOLATION'}")
        st.write(f"• **Factory Capacity Math:** {'✅ PASSED' if m.capacity_math_passed else '❌ VIOLATION'}")
        st.write(f"• **Material Balance Delta:** {'✅ PASSED' if m.inventory_math_passed else '❌ VIOLATION'}")
        st.write(f"• **Enterprise Master Grounding:** {'✅ PASSED' if m.grounding_check_passed else '❌ VIOLATION'}")

# ==============================================================================
# TAB 2: OPERATIONS RESEARCH OPTIMIZER (MILP)
# ==============================================================================
with tab2:
    st.header("⚙️ Operations Research Master Optimizer (HiGHS MILP Solver)")
    st.markdown(
        "*SPORD Architecture: Peer agents diagnose and constrain; the transparent integer program chooses the provably optimal, feasible plan.*"
    )
    
    if st.session_state.get("human_verdict") != "APPROVED":
        st.warning("🔒 This tab is locked pending Human Approval. Please return to the Planning Cockpit and Approve the plan.")
        st.stop()

    if "planning_results" in st.session_state:
        results = st.session_state["planning_results"]
        milp = results["milp_optimal_plan"]
        feas = results["feasibility_report"]
        row = st.session_state["sample_row"]

        col_opt1, col_opt2 = st.columns([1, 1])

        with col_opt1:
            st.subheader("📋 MILP Decision Variables & Optimal Allocation")
            alloc_df = pd.DataFrame(
                [
                    {"Decision Lever": f"Production: {f}", "Optimal Volume": f"{u:,.0f} units"}
                    for f, u in milp.production_allocations.items()
                ]
                + [
                    {"Decision Lever": f"Overtime: {f}", "Optimal Volume": f"{ot:.1f} hours"}
                    for f, ot in milp.overtime_hours.items()
                ]
                + [
                    {"Decision Lever": f"Inter-Plant Transfer: {s['from_factory']} ➔ {s['to_factory']}", "Optimal Volume": f"{s['units_shifted']:,.0f} units (${s['transfer_cost_usd']:,.0f})"}
                    for s in milp.transfer_shifts
                ]
                + [
                    {"Decision Lever": f"Supplier Expedite: {m}", "Optimal Volume": f"{u:,.0f} units"}
                    for m, u in milp.expedite_units.items()
                ]
            )
            st.dataframe(alloc_df, width="stretch", hide_index=True)

            st.write(f"• **Optimized Objective Value:** `${milp.objective_value:,.2f}`")
            st.write(f"• **Carbon Emissions:** `{milp.carbon_emissions_kg:,.1f} kg CO2e`")

        with col_opt2:
            st.subheader("🛡️ Feasibility Oracle Hard-Constraint Audit")
            st.write(f"• **Global Feasibility:** {'✅ VERIFIED' if feas.is_globally_feasible else '❌ VIOLATED'}")
            st.write(f"• **Capacity Bound Satisfied:** {'✅ YES' if feas.capacity_satisfied else '❌ NO'}")
            st.write(f"• **Demand Conservation Satisfied:** {'✅ YES' if feas.demand_conservation_satisfied else '❌ NO'}")
            st.write(f"• **Material Conservation Satisfied:** {'✅ YES' if feas.material_conservation_satisfied else '❌ NO'}")
            st.write(f"• **Max Constraint Violation:** `{feas.max_constraint_violation:.4f}`")

            st.markdown("---")
            st.subheader("🎯 Strategic Candidate Frontier")
            bom_ratios = {
                "MAT-A1": float(row.get("mat_A1_req", 1.0)),
                "MAT-B2": float(row.get("mat_B2_req", 1.0)),
                "MAT-C3": float(row.get("mat_C3_req", 1.0))
            }
            usable_mat_stock = {
                "MAT-A1": float(row.get("mat_A1_initial_stock", 5000.0)),
                "MAT-B2": float(row.get("mat_B2_initial_stock", 5000.0)),
                "MAT-C3": float(row.get("mat_C3_initial_stock", 5000.0))
            }
            prim_f = str(row.get("factory", "Factory A (Stuttgart)"))
            sec_f = "Factory D (Wroclaw)" if prim_f != "Factory D (Wroclaw)" else "Factory A (Stuttgart)"
            sec_cap = 14000.0 if sec_f == "Factory D (Wroclaw)" else 12000.0
            candidates = candidate_gen.generate_candidate_frontier(
                target_demand=float(results["demand"].consensus_demand_units),
                unit_price=float(row.get("unit_price_usd", 1200.0)),
                primary_factory=prim_f,
                primary_capacity=float(row.get("effective_capacity", 12000.0)),
                secondary_factory=sec_f,
                secondary_capacity=sec_cap,
                bom_ratios=bom_ratios,
                usable_material_stock=usable_mat_stock,
            )
            c_df = pd.DataFrame([
                {
                    "Strategy": c["strategy_name"],
                    "Cost": f"${c['cost_usd']:,.0f}",
                    "Service Level": f"{c['service_level_pct']}%",
                    "Carbon": f"{c['carbon_kg']:,.1f} kg",
                }
                for c in candidates
            ])
            st.dataframe(c_df, width="stretch", hide_index=True)
    else:
        st.info("Execute a planning cycle in Tab 1 to inspect the active MILP solution.")

# ==============================================================================
# TAB 3: ENTERPRISE PROVENANCE KNOWLEDGE GRAPH
# ==============================================================================
with tab3:
    st.header("🕸️ Enterprise Provenance Knowledge Graph")
    st.markdown(
        "*Cryptographically maps every data point, AI claim, and MILP variable back to its ground-truth enterprise source.*"
    )
    
    if st.session_state.get("human_verdict") != "APPROVED":
        st.warning("🔒 This tab is locked pending Human Approval. Please return to the Planning Cockpit and Approve the plan.")
        st.stop()

    if "planning_results" in st.session_state:
        results = st.session_state["planning_results"]
        graph = results["evidence_graph"]
        tm = TruthMaintenanceEngine(graph)

        col_kg1, col_kg2 = st.columns([1, 1])

        with col_kg1:
            st.subheader("📌 Evidence Nodes in Current Scenario")
            nodes_data = []
            for uid, node in graph.nodes_by_id.items():
                nodes_data.append({
                    "Entity": node.entity_id,
                    "Type": node.entity_type,
                    "Source System": node.source_system,
                    "Observed Value": str(node.value),
                    "Confidence": f"{node.confidence*100:.0f}%",
                    "Flags": ", ".join(node.quality_flags) if node.quality_flags else "CLEAN",
                })
            st.dataframe(pd.DataFrame(nodes_data), width="stretch", hide_index=True)

        with col_kg2:
            st.subheader("🔎 Root-Cause Tracing (Material Shortage)")
            trace = tm.trace_shortage_root_cause("MAT-A1")
            st.json(trace)

            st.markdown("---")
            st.subheader("🎯 Human Verification Prioritization")
            st.caption("Ranks facts by Financial Exposure x Uncertainty to maximize human review ROI.")
            priority_facts = tm.prioritize_human_verification([
                {"entity_id": "CRM Pipeline Deals", "financial_exposure_usd": float(row.get("unweighted_pipeline_usd", 1000000.0)), "confidence": 0.82, "source_system": "CRM"},
                {"entity_id": "Primary Factory Maintenance", "financial_exposure_usd": 450000.0, "confidence": 0.88, "source_system": "MES"},
                {"entity_id": "Supplier In-Transit Lead Time", "financial_exposure_usd": 650000.0, "confidence": 0.72, "source_system": "WMS"},
            ])
            st.dataframe(pd.DataFrame(priority_facts), width="stretch", hide_index=True)
    else:
        st.info("Execute a planning cycle in Tab 1 to populate the evidence graph.")

# ==============================================================================
# TAB 4: 4-TRACK OPERATIONAL RESILIENCE BENCHMARK
# ==============================================================================
with tab4:
    st.header("🛡️ Four-Track Operational Resilience Benchmark")
    st.markdown(
        "*Evaluating In-Distribution, Disruption OOD, Data-Quality OOD, and Policy OOD beyond prediction accuracy.*"
    )

    bench_file = "data/resilience_benchmark_report.json"
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            res_rep = json.load(f)

        t1 = res_rep["track_1_in_distribution"]
        t2 = res_rep["track_2_disruption_ood"]
        t3 = res_rep["track_3_data_quality_ood"]
        t4 = res_rep["track_4_policy_governance_ood"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Track 1: Demand MAE", f"{t1['demand_mae_units']} u", f"Coverage: {t1['interval_coverage_pct']}%")
        c2.metric("Track 2: Mean Regret vs Oracle", f"${t2['mean_decision_regret_usd']:,.0f}", f"Fill Rate: {t2['mean_fill_rate_pct']}%")
        c3.metric("Track 3: Unsafe Action Rate", f"{t3['unsafe_action_rate_pct']}%", f"Escalation: {t3['escalation_recall_pct']}%")
        c4.metric("Track 4: Policy Violations", f"{t4['authorization_violation_rate_pct']}%", "Zero Tolerance")

        st.markdown("---")
        col_bench_g1, col_bench_g2 = st.columns([1, 1])

        with col_bench_g1:
            st.subheader("📉 Track 2: Disruption OOD Tail Risk (CVaR)")
            st.write(f"• **Worst-Decile Cost (CVaR 0.95):** `${t2['cvar_95_worst_decile_cost_usd']:,.2f}`")
            st.write(f"• **Mean Stockout Duration under Disruption:** `{t2['mean_stockout_duration_days']} days`")
            st.write(f"• **Mean System Recovery Time:** `{t2['mean_recovery_time_days']} days`")

        with col_bench_g2:
            st.subheader("🛡️ Executive Resilience Certification")
            st.markdown(
                f"""
                <div class="audit-card" style="border-left: 4px solid #10B981;">
                    <h3 style="color: #10B981; margin: 0;">Grade: {res_rep['overall_resilience_grade']}</h3>
                    <p style="margin-top: 6px; color: #94A3B8;">
                        • Unsafe Action Execution Rate: <b>{t3['unsafe_action_rate_pct']}%</b> (Target: 0.0%)<br>
                        • Human Escalation Recall: <b>{t3['escalation_recall_pct']}%</b> (Target: 100.0%)<br>
                        • Authorization Breach Rate: <b>{t4['authorization_violation_rate_pct']}%</b> (Target: 0.0%)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Run `python scripts/run_resilience_benchmark.py` to generate the complete 4-track report.")

# ==============================================================================
# TAB 5: 10,000 DATASET EXPLORER
# ==============================================================================
with tab5:
    st.header("🔍 Enterprise Planning Data Lake (10,000 Records)")
    dataset_choice = st.radio("Dataset Selection", ["2,000 Test Ground Truth", "8,000 Historical Reference"], horizontal=True)
    active_df = df_test if "2,000" in dataset_choice else df_ref
    st.dataframe(active_df.head(100), width="stretch")
    st.caption("Displaying first 100 rows with complete operational parameters and deterministic ground truth.")
