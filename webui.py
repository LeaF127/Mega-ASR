"""
语音识别控制台 - UI 框架版
左侧：控制台（音频源选择、参数、录音按钮）
右侧：转写记录区
"""

import streamlit as st
from datetime import datetime
import time
import random

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="语音识别控制台",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# 样式：简约、整齐、大气
# 深底 + 极淡的网格线 + 高级灰文字 + 一个克制的强调色
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif;
    }

    /* 隐藏 streamlit 默认装饰 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 整体留白 */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* 大标题 */
    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #8a8a8a;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* 分区标题 */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #9a9a9a;
        margin-bottom: 0.75rem;
        margin-top: 1.2rem;
    }

    /* 卡片容器 */
    .panel {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 6px;
        padding: 1.5rem 1.5rem;
    }

    /* 状态徽章 */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
        color: #555;
        padding: 0.3rem 0.7rem;
        background: #f7f7f7;
        border-radius: 999px;
        border: 1px solid #ececec;
    }
    .dot {
        width: 6px; height: 6px; border-radius: 50%;
        display: inline-block;
        background: #c0c0c0;
    }
    .dot.live { background: #d64545; box-shadow: 0 0 0 0 rgba(214,69,69,.6); animation: pulse 1.4s infinite; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0   rgba(214,69,69,.5); }
        70%  { box-shadow: 0 0 0 8px rgba(214,69,69,0);   }
        100% { box-shadow: 0 0 0 0   rgba(214,69,69,0);   }
    }

    /* 转写条目 */
    .entry {
        padding: 1rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .entry:last-child { border-bottom: none; }
    .entry-meta {
        font-size: 0.72rem;
        color: #a8a8a8;
        letter-spacing: 0.04em;
        font-variant-numeric: tabular-nums;
        margin-bottom: 0.35rem;
    }
    .entry-text {
        font-size: 0.98rem;
        color: #1f1f1f;
        line-height: 1.65;
    }

    /* 空状态 */
    .empty {
        text-align: center;
        padding: 4rem 1rem;
        color: #bdbdbd;
        font-size: 0.9rem;
    }
    .empty-mark {
        font-size: 2rem;
        color: #e0e0e0;
        margin-bottom: 0.8rem;
        font-weight: 200;
    }

    /* 按钮 */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: 1px solid #1a1a1a;
        background: #1a1a1a;
        color: #fafafa;
        font-weight: 500;
        letter-spacing: 0.04em;
        padding: 0.55rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #333;
        border-color: #333;
        color: #fff;
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }

    /* 次要按钮 */
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        background: #ffffff;
        color: #1a1a1a;
        border: 1px solid #d8d8d8;
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
        background: #f5f5f5;
        border-color: #1a1a1a;
    }

    /* 输入控件细化 */
    .stSelectbox label, .stSlider label, .stRadio label {
        font-size: 0.78rem !important;
        color: #555 !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em;
    }

    /* 统计数字 */
    .stat {
        text-align: left;
    }
    .stat-num {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #9a9a9a;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    /* 分隔线 */
    hr {
        border: none;
        border-top: 1px solid #ececec;
        margin: 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 状态初始化
# ──────────────────────────────────────────────
if "records" not in st.session_state:
    st.session_state.records = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# 模拟转写文本池
MOCK_TEXTS = [
    "今天的会议主要讨论了第四季度的产品路线图。",
    "我们需要在本月底之前完成原型设计。",
    "客户反馈表明，界面的响应速度仍有提升空间。",
    "下一步计划是在两周内启动用户测试。",
    "请把这份文档转发给产品和设计团队。",
    "服务器在高并发场景下的稳定性已通过验证。",
    "整体进度比预期提前了大约三天。",
    "建议把这一项移到下一个迭代再处理。",
]


def add_mock_record(source: str):
    """添加一条模拟转写记录"""
    st.session_state.records.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": source,
            "text": random.choice(MOCK_TEXTS),
        },
    )


# ──────────────────────────────────────────────
# 标题
# ──────────────────────────────────────────────
st.markdown('<div class="app-title">语音识别控制台</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Speech Recognition Console</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 主体两栏布局
# ──────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

# ============== 左栏：控制台 ==============
with left:
    with st.container(border=False):
        # 状态指示
        status_dot = "live" if st.session_state.is_recording else ""
        status_label = "录音中" if st.session_state.is_recording else "待机"
        st.markdown(
            f'<div class="status-pill"><span class="dot {status_dot}"></span>{status_label}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">音频源</div>', unsafe_allow_html=True)
        source = st.selectbox(
            "音频源",
            ["麦克风", "音频文件", "系统音频"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-label">识别语言</div>', unsafe_allow_html=True)
        language = st.selectbox(
            "识别语言",
            ["中文（普通话）", "English", "中英混合", "粤语", "日本語"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-label">模型</div>', unsafe_allow_html=True)
        model = st.selectbox(
            "模型",
            ["Whisper Large v3", "Whisper Medium", "Paraformer", "SenseVoice"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-label">灵敏度</div>', unsafe_allow_html=True)
        sensitivity = st.slider(
            "灵敏度", 0, 100, 65, label_visibility="collapsed"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # 操作按钮
        c1, c2 = st.columns(2)
        with c1:
            btn_label = "停止" if st.session_state.is_recording else "开始录音"
            if st.button(btn_label, use_container_width=True):
                if st.session_state.is_recording:
                    # 停止：追加一条模拟记录
                    add_mock_record(source)
                    st.session_state.is_recording = False
                else:
                    st.session_state.is_recording = True
                st.rerun()
        with c2:
            if st.button("清空记录", use_container_width=True):
                st.session_state.records = []
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # 统计
        s1, s2 = st.columns(2)
        total = len(st.session_state.records)
        chars = sum(len(r["text"]) for r in st.session_state.records)
        with s1:
            st.markdown(
                f'<div class="stat"><div class="stat-num">{total}</div>'
                f'<div class="stat-label">条记录</div></div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f'<div class="stat"><div class="stat-num">{chars}</div>'
                f'<div class="stat-label">字符</div></div>',
                unsafe_allow_html=True,
            )

# ============== 右栏：记录区 ==============
with right:
    st.markdown(
        '<div class="section-label" style="margin-top:0;">转写记录</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.records:
        st.markdown(
            '<div class="panel"><div class="empty">'
            '<div class="empty-mark">◌</div>'
            '暂无记录 · 点击左侧"开始录音"以生成示例'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        html = '<div class="panel">'
        for r in st.session_state.records:
            html += (
                f'<div class="entry">'
                f'<div class="entry-meta">{r["date"]} · {r["time"]} · {r["source"]}</div>'
                f'<div class="entry-text">{r["text"]}</div>'
                f"</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 录音中：模拟实时转写（每 ~2 秒追加一条）
# ──────────────────────────────────────────────
if st.session_state.is_recording:
    time.sleep(2)
    add_mock_record(source)
    st.rerun()