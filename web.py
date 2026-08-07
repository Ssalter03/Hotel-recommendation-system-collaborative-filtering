import streamlit as st
import pandas as pd
import unicodedata
import traceback
import joblib
import numpy as np
import base64
import sys
import os
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql import SparkSession
import plotly.graph_objects as go
import plotly.express as px

# ===== ĐẢM BẢO PYTHON NHẬN DIỆN THƯ MỤC FUNCTIONS =====
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from functions.hotel_insight import (
        generate_hotel_report,
        plot_hotel_radar_new,
        plot_score_difference_new,
        generate_score_difference_insight,
        plot_customer_type_pie,
        plot_nationality_pie,
        plot_nationality_pie_vn,
        plot_group_name_pie,
        plot_group_analysis,
        plot_group_bubble_chart,
        plot_review_trend_by_month,
        plot_review_trend_by_year
    )
except ImportError:
    pass

# =========================================================
# CẤU HÌNH TRANG WEB CHÍNH & LOAD ẢNH THEO ĐƯỜNG DẪN TUYỆT ĐỐI
# =========================================================
st.set_page_config(
    page_title="Hệ Thống Gợi Ý Khách Sạn Sử Dụng Model ALS Collaborative Filtering",
    page_icon="🏨",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

def get_image_uri(filename):
    img_path = BASE_DIR / filename
    if img_path.exists():
        return "data:image/jpeg;base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")
    return ""

content_based_cover_data_uri = get_image_uri("content-based_cover.jpg")
home_page_cover_data_uri = get_image_uri("HomePage_cover.jpg")
recommend_cover_uri = get_image_uri("recommend_cover.jpg")
personalized_cover_uri = get_image_uri("personalized_cover.jpg")
explore_cover_uri = get_image_uri("explore_cover.jpg")

st.markdown(
    f"""
    <style>
        .stText, p, span {{
            font-size: 20px !important;
        }}
        .stCaptionText, caption {{
            font-size: 16px !important;
        }}
        [data-testid="stSidebar"] {{
            font-size: 14px !important;
            text-align: left !important;
        }}
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            font-size: 14px !important;
            text-align: left !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            padding: 0.45rem 0.7rem !important;
            min-height: 2rem !important;
            text-align: left !important;
        }}
        .sidebar-title {{
            font-size: 1.55rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            margin: 0 0 0.75rem 0 !important;
            text-align: left !important;
        }}
        .sidebar-section-label {{
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            margin: 0.85rem 0 0.35rem 0 !important;
            text-align: left !important;
        }}
        .sidebar-project-card {{
            background: var(--secondary-background-color);
            border: 1px solid rgba(87, 160, 241, 0.42);
            border-radius: 16px;
            padding: 0.9rem 0.95rem;
            margin-top: 0.6rem;
            text-align: left !important;
            line-height: 1.5 !important;
        }}
        .hero-bubble {{
            position: relative;
            min-height: 360px;
            border: 1px solid rgba(128, 194, 255, 0.55);
            border-radius: 26px;
            padding: 1rem 1.2rem;
            margin: 0.2rem 0 1rem 0;
            box-shadow: 0 12px 30px rgba(2, 12, 33, 0.24);
            overflow: hidden;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            text-align: center;
            background:
                linear-gradient(90deg, rgba(7, 28, 51, 0.74) 0%, rgba(7, 28, 51, 0.45) 50%, rgba(7, 28, 51, 0.68) 100%),
                url("{home_page_cover_data_uri}") center/cover no-repeat;
        }}
        .hero-title {{
            position: relative;
            z-index: 1;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            line-height: 1.05 !important;
            margin: 0 !important;
            color: #ffffff !important;
            text-align: center !important;
        }}
        .hero-subtitle {{
            position: relative;
            z-index: 1;
            font-size: 1rem !important;
            color: #e9f5ff !important;
            margin-top: 0.45rem !important;
            line-height: 1.4 !important;
            text-align: center !important;
        }}
        .section-pill {{
            display: inline-block;
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(128, 194, 255, 0.55);
            border-radius: 14px;
            padding: 0.45rem 0.9rem;
            margin: 0 0 0.9rem 0;
            font-size: 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }}
        .quick-action-card {{
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(87, 160, 241, 0.42);
            border-radius: 18px;
            padding: 1rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.8rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        }}
        .quick-action-title {{
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            color: var(--text-color);
            margin: 0 !important;
        }}
        .quick-action-body {{
            font-size: 0.98rem !important;
            line-height: 1.5 !important;
            color: var(--text-color);
            opacity: 0.9;
            margin: 0 !important;
        }}
        .section-separator {{
            height: 1px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(45, 130, 221, 0.85), rgba(45, 130, 221, 0.15));
            margin: 1rem 0 0.9rem 0;
        }}
        .similarity-hero {{
            position: relative;
            min-height: 420px;
            border-radius: 26px;
            overflow: hidden;
            padding: 2rem 1.4rem 1.4rem 1.4rem;
            display: flex;
            align-items: flex-end;
            justify-content: flex-start;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
            margin: 0.4rem 0 0.8rem 0;
        }}
        .similarity-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.14);
        }}
        .similarity-hero-content {{
            position: relative;
            z-index: 1;
            color: #ffffff;
            max-width: 920px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 1.3rem;
            padding-top: 2rem;
        }}
        .similarity-hero h1 {{
            margin: 0 !important;
            color: #ffffff !important;
            font-size: clamp(2.5rem, 6.6vw, 5.2rem) !important;
            line-height: 1.02 !important;
            font-weight: 900 !important;
            letter-spacing: 0.02em !important;
            word-spacing: 0.01em !important;
            text-shadow: 0 3px 10px rgba(0, 0, 0, 0.75), 0 0 18px rgba(12, 61, 108, 0.45);
            word-break: break-word;
        }}
        .similarity-hero p {{
            margin: 0 !important;
            color: #f7fbff !important;
            font-size: 1.08rem !important;
            line-height: 1.55 !important;
            max-width: 860px;
            text-shadow: 0 2px 14px rgba(0, 0, 0, 0.45);
        }}

        /* FIX NÚT OK LƠ LỬNG PHÍA DƯỚI BÊN PHẢI MÀN HÌNH */
        button[kind="primary"] {{
            position: fixed !important;
            bottom: 20px !important;
            right: 30px !important;
            z-index: 9999 !important;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5) !important;
            border-radius: 30px !important;
            padding: 12px 28px !important;
            font-weight: bold !important;
            transition: 0.3s;
        }}
        button[kind="primary"]:hover {{
            transform: scale(1.05);
        }}

        /* KHUNG CẤU HÌNH LƠ LỬNG CỐ ĐỊNH PHÍA TRÊN NÚT OK */
        div[data-key="floating_config_container"],
        div[class*="st-key-floating_config_container"] {{
            position: fixed !important;
            bottom: 85px !important;
            right: 30px !important;
            width: 450px !important;
            max-width: 90vw !important;
            z-index: 9998 !important;
            background-color: var(--secondary-background-color, #1e222d) !important;
            border: 1px solid rgba(87, 160, 241, 0.6) !important;
            border-radius: 16px !important;
            padding: 2px !important;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.5) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== HÀM HIỂN THỊ HERO BANNER DÙNG CHUNG =====
def render_hero_banner(title: str, subtitle: str, cover_uri: str = "", bg_position: str = "center"):
    if cover_uri:
        bg_style = f"background: linear-gradient(90deg, rgba(7, 28, 51, 0.78) 0%, rgba(7, 28, 51, 0.42) 50%, rgba(7, 28, 51, 0.58) 100%), url('{cover_uri}') {bg_position}/cover no-repeat;"
    else:
        bg_style = "background: linear-gradient(90deg, rgba(7, 28, 51, 0.95) 0%, rgba(7, 28, 51, 0.8) 100%);"

    st.markdown(
        f'''
        <div class="similarity-hero" style="{bg_style}">
            <div class="similarity-hero-content">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

# ===== HÀM BỔ TRỢ & CHUẨN HÓA DỮ LIỆU =====
def normalize_text(s):
    if pd.isna(s):
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

def format_number(value, decimals=2, fallback="N/A"):
    if value is None or pd.isna(value):
        return fallback
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned: return fallback
        if cleaned.count(",") > 0 and cleaned.count(".") == 0:
            cleaned = cleaned.replace(",", ".")
        elif cleaned.count(",") > 0:
            cleaned = cleaned.replace(",", "")
        try:
            number = float(cleaned)
        except (TypeError, ValueError):
            return cleaned
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
    if decimals is None:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}"

pd.options.display.float_format = lambda x: format_number(x, decimals=2)

def get_short_description(text, max_len=180):
    if pd.isna(text):
        return "Đang cập nhật"
    s = str(text).strip().replace("\n", " ")
    if len(s) > max_len:
        return s[:max_len].rsplit(' ', 1)[0] + "..."
    return s

def get_second_paragraph(text):
    if pd.isna(text):
        return "Đang cập nhật"
    text = str(text).strip()
    if not text:
        return "Đang cập nhật"
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs[1]
    if paragraphs:
        return paragraphs[0]
    return text

def render_compact_hotel_details(hotel_row):
    description = get_second_paragraph(hotel_row.get("Hotel_Description", "Đang cập nhật"))
    cleanliness = format_number(hotel_row.get("Cleanliness", "N/A"), decimals=2)
    services = format_number(hotel_row.get("Service", hotel_row.get("Services", "N/A")), decimals=2)
    value_for_money = format_number(hotel_row.get("Value_for_money", "N/A"), decimals=2)

    st.markdown(f"**Mô tả:** {description}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🧼 Điểm sạch sẽ: {cleanliness}")
    with col2:
        st.caption(f"🛎️ Dịch vụ: {services}")
    with col3:
        st.caption(f"💰 Điểm đáng tiền: {value_for_money}")

def render_points_only_hotel_details(hotel_row):
    cleanliness = format_number(hotel_row.get("Cleanliness", "N/A"), decimals=2)
    services = format_number(hotel_row.get("Service", hotel_row.get("Services", "N/A")), decimals=2)
    value_for_money = format_number(hotel_row.get("Value_for_money", "N/A"), decimals=2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🧼 Điểm sạch sẽ: {cleanliness}")
    with col2:
        st.caption(f"🛎️ Dịch vụ: {services}")
    with col3:
        st.caption(f"💰 Điểm đáng tiền: {value_for_money}")

def render_star_ratings(row):
    metrics = [
        ("📍 Vị trí (Location)", "Location"),
        ("🧼 Sạch sẽ (Cleanliness)", "Cleanliness"),
        ("🛎️ Dịch vụ (Service)", "Service"),
        ("🏊 Cơ sở vật chất (Facilities)", "Facilities"),
        ("💰 Giá trị mang lại (Value for money)", "Value_for_money"),
        ("🛏️ Thoải mái & Phòng (Comfort & room quality)", "Comfort_and_room_quality")
    ]
    
    def score_to_stars(val):
        if pd.isna(val) or val is None or str(val).strip() in ["", "N/A", "nan"]:
            return "N/A"
        try:
            score_10 = float(str(val).replace(",", "."))
            score_5 = max(0.0, min(5.0, score_10 / 2.0))
            full_stars = int(round(score_5))
            star_str = "★" * full_stars + "☆" * (5 - full_stars)
            return f"{star_str} ({score_5:.1f}/5)"
        except Exception:
            return "N/A"

    st.markdown("####  Đánh giá chi tiết (Thang điểm 0 - 5 ⭐):")
    cols = st.columns(2)
    for idx, (label, key) in enumerate(metrics):
        val = row.get(key, row.get("Services" if key == "Service" else key, "N/A"))
        star_display = score_to_stars(val)
        with cols[idx % 2]:
            st.markdown(f"- **{label}:** {star_display}")
            
    comments_cnt = row.get("comments_count", row.get("Comments_Count", None))
    
    is_invalid_or_zero = False
    if comments_cnt is None or pd.isna(comments_cnt) or str(comments_cnt).strip() in ["", "N/A", "nan", "None"]:
        is_invalid_or_zero = True
    else:
        try:
            cnt_val = int(float(str(comments_cnt).replace(",", "")))
            if cnt_val <= 0:
                is_invalid_or_zero = True
            else:
                comments_cnt_fmt = f"{cnt_val:,} lượt"
        except Exception:
            is_invalid_or_zero = True

    if is_invalid_or_zero:
        st.markdown("💬 **Số lượng đánh giá (Reviews):** *Không có thông tin*")
    else:
        st.markdown(f"💬 **Số lượng đánh giá (Reviews):** `{comments_cnt_fmt}`")
        
    st.markdown("---")

# ===== LOAD DỮ LIỆU CƠ BẢN =====
@st.cache_data
def load_base_data():
    try:
        hotel_info = pd.read_csv("data/hotel_info.csv")
        hotel_info["Address_norm"] = hotel_info["Hotel_Address"].apply(normalize_text)
        hotel_info["Description_norm"] = hotel_info["Hotel_Description"].apply(normalize_text)
        hotel_info["Hotel_ID"] = hotel_info["Hotel_ID"].astype(str).str.strip()
        return hotel_info
    except Exception as e:
        st.error(f"Lỗi khi đọc file hotel_info.csv: {e}")
        return None

hotel_info = load_base_data()

# ===== LOAD DỮ LIỆU DÀNH RIÊNG CHO HOTEL INSIGHT REPORT =====
@st.cache_data
def load_hotel_insight_info():
    try:
        return pd.read_csv("data/hotel_info_cleaned.csv")
    except Exception:
        return None

@st.cache_data
def load_hotel_insight_comments():
    try:
        return pd.read_csv("data/hotel_comments_cleaned.csv")
    except Exception:
        return None

@st.cache_data
def load_model_comparison_data():
    try: return pd.read_csv("gensim_vs_cosine_comparison.csv")
    except Exception: return None

@st.cache_data
def load_review_data():
    try: return pd.read_csv("data/hotel_reviews.csv")
    except Exception: return None

comparison_df = load_model_comparison_data()
review_df = load_review_data()

@st.cache_data
def load_cosine_model():
    try: return joblib.load("cosine_similarity_model.joblib")
    except Exception: return None

cosine_model = load_cosine_model()

def render_section_title(title, icon=""):
    st.markdown(f'<div class="section-pill">{icon} {title}</div>', unsafe_allow_html=True)

def get_similarity_recommendations(selected_hotel_name, recommended_count=5, exclude_names=None):
    if cosine_model is None: return None
    data_search = cosine_model.get("data_search")
    combined_text = cosine_model.get("combined_text")
    vectorizer = cosine_model.get("vectorizer")
    tfidf_matrix = cosine_model.get("tfidf_matrix")
    if data_search is None or combined_text is None or vectorizer is None or tfidf_matrix is None:
        return None

    data_search = data_search.copy()
    data_search["Hotel_Name_norm"] = data_search["Hotel_Name"].fillna("").astype(str).str.strip().str.lower()
    selected_norm = str(selected_hotel_name).strip().lower()
    matching_rows = data_search[data_search["Hotel_Name_norm"] == selected_norm]
    if matching_rows.empty: return None

    selected_idx = int(matching_rows.index[0])
    selected_text = combined_text[selected_idx] if selected_idx < len(combined_text) else ""
    if not selected_text: return None

    selected_vector = vectorizer.transform([selected_text])
    similarity_scores = cosine_similarity(selected_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_indices = [idx for idx in ranked_indices if idx != selected_idx]

    if exclude_names:
        exclude_set = {str(name).strip().lower() for name in exclude_names if str(name).strip()}
        ranked_indices = [idx for idx in ranked_indices if str(data_search.iloc[idx]["Hotel_Name"]).strip().lower() not in exclude_set]

    top_indices = ranked_indices[:recommended_count]
    recs = data_search.iloc[top_indices].copy()
    recs["similarity_score"] = [float(similarity_scores[idx]) for idx in top_indices]
    recs = recs.sort_values(by="similarity_score", ascending=False)

    display_df = recs.merge(
        hotel_info[["Hotel_ID", "Hotel_Name", "Hotel_Address", "Hotel_Rank", "Total_Score", "Hotel_Description", "Location", "Cleanliness", "Service", "Facilities", "Value_for_money", "Comfort_and_room_quality", "comments_count"]].copy(),
        on="Hotel_Name",
        how="left",
        suffixes=("", "_info")
    )
    return display_df

def get_vietnam_cities(df):
    if df is None or df.empty:
        return ["Nha Trang", "Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Vũng Tàu", "Đà Lạt", "Phú Quốc"]
    popular_cities = {
        "Nha Trang": ["nha trang", "nhatrang"],
        "Hồ Chí Minh": ["ho chi minh", "hcm", "sài gòn", "sai gon"],
        "Hà Nội": ["ha noi", "hanoi"],
        "Đà Nẵng": ["da nang", "danang"],
        "Vũng Tàu": ["vung tau", "vungtau"],
        "Đà Lạt": ["da lat", "dalat"],
        "Phú Quốc": ["phu quoc", "phuquoc"]
    }
    available_cities = []
    sample_addresses = df["Address_norm"].str.cat(sep=" ")
    for city_name, keywords in popular_cities.items():
        if any(kw in sample_addresses for kw in keywords):
            available_cities.append(city_name)
    return available_cities if available_cities else ["Nha Trang", "Hồ Chí Minh", "Hà Nội"]

cities_list = get_vietnam_cities(hotel_info)

# Spark Session Initialization
if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("HotelRecsStreamlit") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
        
if "hotel_indexer_model" not in st.session_state:
    try: st.session_state.hotel_indexer_model = StringIndexerModel.load("models/hotel_indexer")
    except Exception as e: st.error(f"Lỗi load hotel_indexer: {e}")

if "als_hotel_model" not in st.session_state:
    try: st.session_state.als_hotel_model = ALSModel.load("models/als_hotel_model")
    except Exception as e: st.error(f"Lỗi load als_hotel_model: {e}")

# Session State Manager
if "show_recommendations" not in st.session_state: st.session_state["show_recommendations"] = False
if "recs_result_data" not in st.session_state: st.session_state["recs_result_data"] = None
if "chosen_hotels_data" not in st.session_state: st.session_state["chosen_hotels_data"] = None
if "show_limit" not in st.session_state: st.session_state["show_limit"] = 20
if "matched_hotels" not in st.session_state: st.session_state["matched_hotels"] = None
if "search_keywords" not in st.session_state: st.session_state["search_keywords"] = ""
if "is_strict_street" not in st.session_state: st.session_state["is_strict_street"] = False

# =========================================================
# MENU ĐIỀU HƯỚNG BÊN TRÁI GUI
# =========================================================
st.sidebar.markdown("<div class='sidebar-title'>🧭 HỆ THỐNG MENU</div>", unsafe_allow_html=True)

if "pending_home_nav" not in st.session_state: st.session_state["pending_home_nav"] = None
if st.session_state.get("pending_home_nav"):
    st.session_state["menu_selection_key"] = st.session_state["pending_home_nav"]
    st.session_state["pending_home_nav"] = None

if "menu_selection_key" not in st.session_state:
    st.session_state["menu_selection_key"] = "🏠 Trang chủ"

def render_nav_button(label: str, key_suffix: str):
    button_key = f"nav_{key_suffix.replace(' ', '_').replace('✨', 'spark').replace('/', '_').replace('📊', 'report').replace('👥', 'team').replace('🎯', 'recommend').replace('🏨', 'hotel').replace('🧠', 'similarity').replace('🏠', 'home')}"
    if st.sidebar.button(label, key=button_key, use_container_width=True):
        st.session_state["menu_selection_key"] = label

render_nav_button("🏠 Trang chủ", "home")
st.sidebar.markdown("<div class='sidebar-section-label'>Các chức năng</div>", unsafe_allow_html=True)

render_nav_button("🧠 Tìm khách sạn tương đồng", "🧠 Tìm khách sạn tương đồng")
render_nav_button("🎯 Gợi ý khách sạn", "🎯 Gợi ý khách sạn")

if st.session_state["show_recommendations"]:
    render_nav_button("✨ Personalized Results", "personalized_results")

render_nav_button("🏨 Khám phá thông tin khách sạn", "🏨 Khám phá thông tin khách sạn")
render_nav_button("📊 Hotel Insight Report", "📊 Hotel Insight Report")

st.sidebar.markdown("<div class='sidebar-section-label'>Thông tin về dự án</div>", unsafe_allow_html=True)
for item in ["📊 So sánh, đánh giá các models", "👥 Bảng phân công việc"]:
    render_nav_button(item, item)

menu_selection = st.session_state["menu_selection_key"]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-project-card">
        <div>👥 <strong>Nhóm thực hiện:</strong></div>
        <div>- Huỳnh Bửu Khang</div>
        <div>- Phạm Tuấn Kiệt</div>
        <div>- Nguyễn Hữu Nguyên Khôi</div>
        <div>🎓 <strong>Đồ án tốt nghiệp Data Science</strong></div>
        <div>Topic: Recommender System</div>
        <div>📅 <strong>Thời điểm thiết kế:</strong> 08/2026</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MỤC 0: TRANG CHỦ
# =========================================================
if menu_selection == "🏠 Trang chủ":
    st.markdown(
        """
        <div class="hero-bubble">
            <div style="display:flex; align-items:flex-end; justify-content:center; flex-wrap: wrap; position: relative; z-index: 1; width: 100%; min-height: 100%; padding-bottom: 0rem;">
                <div style="width: 100%;">
                    <div class="hero-title">Hotel Advisor</div>
                    <div class="hero-subtitle">Hệ thống gợi ý khách sạn & phân tích insight · Dựa trên dữ liệu Agoda thực tế tại Nha Trang, Việt Nam</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("Thao tác nhanh", "⚡")
    quick_actions = [
        ("🧠 Tìm khách sạn tương đồng", "Khám phá khách sạn có mô tả tương đồng nhất bằng cosine similarity.", "🧠 Tìm khách sạn tương đồng"),
        ("🎯 Gợi ý theo sở thích", "Lọc theo thành phố/đường/đặc điểm rồi nhận đề xuất cá nhân hóa từ mô hình ALS.", "🎯 Gợi ý khách sạn"),
        ("🏨 Xem insights về khách sạn", "Xem chi tiết điểm sạch sẽ, dịch vụ, giá trị tiền bạc và mô tả khách sạn.", "🏨 Khám phá thông tin khách sạn"),
        ("📊 Hotel Insight Report", "Phân tích chuyên sâu về hiệu suất vận hành, tập khách hàng và xu hướng review.", "📊 Hotel Insight Report"),
    ]

    action_cols = st.columns(len(quick_actions))
    for index, (title, body, target_menu) in enumerate(quick_actions):
        with action_cols[index]:
            with st.container(border=True):
                st.markdown(f'<div class="quick-action-card"><div class="quick-action-title">{title}</div><div class="quick-action-body">{body}</div></div>', unsafe_allow_html=True)
                if st.button("Mở chức năng", key=f"quick_action_{index}", use_container_width=True):
                    st.session_state["pending_home_nav"] = target_menu
                    st.rerun()

    st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
    render_section_title("Tổng quan dữ liệu", "📌")
    
    if hotel_info is not None and review_df is not None:
        total_hotels = hotel_info["Hotel_ID"].dropna().astype(str).str.strip().nunique()
        total_comments = int(review_df["comments_count"].fillna(0).astype(float).sum()) if "comments_count" in review_df.columns else int(review_df.shape[0])
        avg_total_score = float(review_df["Total_Score"].mean()) if "Total_Score" in review_df.columns else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("🏨 Tổng số khách sạn", format_number(total_hotels, decimals=None))
        col2.metric("📝 Tổng số đánh giá", format_number(total_comments, decimals=None))
        col3.metric("⭐ Điểm đánh giá trung bình", format_number(avg_total_score, decimals=2))

# =========================================================
# MỤC: HOTEL INSIGHT REPORT
# =========================================================
elif menu_selection == "📊 Hotel Insight Report":
    render_section_title("Hotel Insight Report", "📊")
    
    st.caption(
        """
        Phân tích thông tin dựa trên dữ liệu đánh giá từ Agoda,
        giúp chủ khách sạn có cái nhìn tổng quan về cách khách hàng
        nhìn nhận khách sạn, từ đó có chiến lược phát triển kinh doanh phù hợp.
        """
    )

    insight_info = load_hotel_insight_info()
    insight_comments = load_hotel_insight_comments()

    if insight_info is not None and insight_comments is not None:
        hotel_list = sorted(insight_info["Hotel_Name"].dropna().unique())
        selected_hotel = st.selectbox("Chọn khách sạn phân tích:", hotel_list)

        if st.button("🚀 Generate Report", type="primary"):
            st.divider()

            score_cols = ["Location", "Cleanliness", "Service", "Facilities", "Value_for_money"]

            try:
                report = generate_hotel_report(insight_info, selected_hotel, score_cols)
                overview = report["overview"]
                reliability = report["reliability"]

                st.header(f"🌸 {selected_hotel} 🌸")
                st.caption(f"📍 {overview['Address']}")

                # KPI Metrics
                raw_rank = str(overview.get("Rank (Star)", ""))
                matches = re.findall(r'\d+', raw_rank)
                
                if matches:
                    rank_num = int(matches[0])
                    stars = "⭐" * rank_num
                else:
                    stars = "Not Available"

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Hạng sao", stars)
                with c2: st.metric("Tổng điểm", f'{overview["Total Score"]:.1f}/10')
                with c3: st.metric("Lượt đánh giá", f'{overview["Comments"]:,}')
                with c4: st.metric("Percentile", f'{report["percentile"]:.1f}%')

                # Performance Level & Reliability
                left, right = st.columns(2)
                with left:
                    with st.container(border=True):
                        st.subheader("💡 Đánh giá về khách sạn")
                        insight = report["insight"]
                        st.metric("Performance Level", insight["level"])

                        if insight["level"] in ["Xuất sắc", "Rất tốt"]: st.success(insight["message"])
                        elif insight["level"] == "Tốt": st.info(insight["message"])
                        elif insight["level"] == "Trung bình": st.warning(insight["message"])
                        else: st.error(insight["message"])

                with right:
                    with st.container(border=True):
                        st.subheader("⭐ Độ tin cậy của dữ liệu")
                        st.metric("Reliability", reliability["reliability"])

                        if reliability["reliability"] == "Cao": st.success(reliability["message"])
                        elif reliability["reliability"] == "Trung bình": st.warning(reliability["message"])
                        else: st.error(reliability["message"])

                st.divider()

                # Analysis Tabs
                tab1, tab2, tab3 = st.tabs([
                    "📊 Overview - Performance Analysis",
                    "👥 Thống kê Khách hàng",
                    "📈 Xu hướng phát triển của KS"
                ])

                with tab1:
                    st.subheader("📊 Performance Analysis")
                    st.markdown(
                        """
                        **Performance Analysis** cung cấp cái nhìn tổng quan về chất lượng vận hành
                        của khách sạn thông qua việc so sánh điểm đánh giá với mức trung bình toàn hệ thống.
                        """
                    )
                    with st.container(border=True):
                        fig_radar = plot_hotel_radar_new(insight_info, selected_hotel, score_cols)
                        fig_score_difference = plot_score_difference_new(insight_info, selected_hotel, score_cols)

                        col1, col2 = st.columns([1.5, 2])
                        with col1: st.plotly_chart(fig_radar, use_container_width=True)
                        with col2: st.plotly_chart(fig_score_difference, use_container_width=True)

                        st.divider()
                        insight_diff = generate_score_difference_insight(insight_info, selected_hotel, score_cols, top_n=5)
                        st.markdown(insight_diff["strength"])
                        st.markdown(insight_diff["weakness"])
                        st.markdown(insight_diff["summary"])

                with tab2:
                    st.subheader("👥 Thống kê Khách hàng")
                    with st.container(border=True):
                        st.markdown("### 🌏 Phân tích xu hướng Quốc nội - Quốc tế")
                        fig_customer_type = plot_customer_type_pie(insight_comments, selected_hotel)
                        st.plotly_chart(fig_customer_type, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("<b>🌍 Phân bố quốc tịch (không bao gồm Việt Nam)</b>", unsafe_allow_html=True)
                            fig_nat = plot_nationality_pie(insight_comments, selected_hotel)
                            st.plotly_chart(fig_nat, use_container_width=True)
                        with col2:
                            st.markdown("<b>🇻🇳 Phân bố quốc tịch (bao gồm Việt Nam)</b>", unsafe_allow_html=True)
                            fig_nat_vn = plot_nationality_pie_vn(insight_comments, selected_hotel)
                            st.plotly_chart(fig_nat_vn, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 👥 Phân tích theo nhóm khách")
                        fig_group_pie = plot_group_name_pie(insight_comments, selected_hotel)
                        st.plotly_chart(fig_group_pie, use_container_width=True)

                        fig_stay, fig_score = plot_group_analysis(insight_comments, selected_hotel)
                        col1, col2 = st.columns(2)
                        with col1: st.plotly_chart(fig_stay, use_container_width=True)
                        with col2: st.plotly_chart(fig_score, use_container_width=True)

                        st.markdown("<b>💎 Guest Value Matrix</b>", unsafe_allow_html=True)
                        fig_bubble = plot_group_bubble_chart(insight_comments, selected_hotel)
                        st.plotly_chart(fig_bubble, use_container_width=True)

                with tab3:
                    st.subheader("📈 Review Trend Analysis")
                    with st.container(border=True):
                        fig_review_month = plot_review_trend_by_month(insight_comments, selected_hotel)
                        fig_review_year = plot_review_trend_by_year(insight_comments, selected_hotel)

                        st.markdown("### 📅 Review Trend by Month")
                        st.plotly_chart(fig_review_month, use_container_width=True)
                        st.divider()
                        st.markdown("### 📈 Review Trend by Year")
                        st.plotly_chart(fig_review_year, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi khi tạo báo cáo insight cho khách sạn: {e}")
                st.code(traceback.format_exc(), language="python")
    else:
        st.warning("Không tìm thấy các file dữ liệu `hotel_info_cleaned.csv` hoặc `hotel_comments_cleaned.csv` trong thư mục `data/`.")

# =========================================================
# MỤC: SO SÁNH MODEL
# =========================================================
elif menu_selection == "📊 So sánh, đánh giá các models":
    render_section_title("Content-based Filtering", "🧠")
    st.markdown("### Quy trình xây dựng model")
    st.markdown("Data overview & preparation → Data cleaning → Exploratory Data Analysis → Modeling → Evaluation → Export")

    st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Không thể tải dữ liệu so sánh mô hình content-based.")

    st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
    render_section_title("Collaborative Filtering", "🤝")
    
    als_summary_df = pd.DataFrame([{
        "Model": "Spark ALS",
        "Rank": 6,
        "Max Iterations": 10,
        "RMSE": 0.189732
    }])
    st.dataframe(als_summary_df, use_container_width=True, hide_index=True)

# =========================================================
# MỤC 3: RECOMMENDATION SYSTEM
# =========================================================
elif menu_selection == "🎯 Gợi ý khách sạn":
    render_section_title("Gợi ý khách sạn", "🎯")
    render_hero_banner(
        title="Gợi ý khách sạn",
        subtitle="Tìm kiếm địa điểm lưu trú lý tưởng theo Tỉnh/Thành phố hoặc Tuyến đường bạn yêu thích.",
        cover_uri=recommend_cover_uri,
        bg_position="center 65%"
    )

    if hotel_info is not None:
        if "has_searched" not in st.session_state: st.session_state["has_searched"] = False
        if "matched_hotels" not in st.session_state: st.session_state["matched_hotels"] = None
        if "search_title" not in st.session_state: st.session_state["search_title"] = ""
        if "city_norm_state" not in st.session_state: st.session_state["city_norm_state"] = ""

        selected_city = st.selectbox(" Chọn Tỉnh / Thành phố:", options=cities_list)
        city_norm = normalize_text(selected_city).strip() 

        street_input = st.text_input(
            "Nhập tên đường (Tùy chọn - Có hoặc không dấu):", 
            placeholder="Ví dụ: Loc Tho, Tran Phu, Le Loi... (Để trống để xem toàn bộ thành phố)"
        )

        if st.button("🔍 Tìm khách sạn"):
            st.session_state["has_searched"] = True
            st.session_state["city_norm_state"] = city_norm
            
            city_condition = hotel_info["Address_norm"].apply(lambda addr: city_norm in str(addr).lower())
            city_hotels = hotel_info[city_condition].copy()

            if city_hotels.empty:
                st.error("❌ Không có khách sạn ở địa điểm này")
                st.session_state["matched_hotels"] = None
                st.session_state["search_title"] = ""
            else:
                matched = city_hotels.copy()
                if street_input.strip():
                    street_norm = normalize_text(street_input).strip()
                    st.session_state["search_keywords"] = street_norm
                    
                    def check_street_smart(addr_norm, street_kw, c_norm):
                        addr_str = str(addr_norm).lower()
                        street_area = addr_str.split(c_norm)[0] if c_norm in addr_str else addr_str
                        return street_kw in street_area

                    street_condition = matched["Address_norm"].apply(
                        lambda addr: check_street_smart(addr, street_norm, city_norm)
                    )
                    matched = matched[street_condition].copy()
                else:
                    st.session_state["search_keywords"] = ""

                if matched.empty:
                    st.session_state["matched_hotels"] = None
                    st.session_state["search_title"] = f'📭 Không tìm thấy khách sạn nào phù hợp tại {selected_city}.'
                else:
                    title_parts = [f"tại {selected_city}"]
                    if street_input.strip():
                        title_parts.append(f'trên đường "{street_input.strip()}"')
                    
                    st.session_state["search_title"] = f"📌 Danh sách khách sạn phù hợp {' '.join(title_parts)}:"
                    if "Total_Score" in matched.columns:
                        matched["Total_Score"] = matched["Total_Score"].fillna(0.0)
                    st.session_state["matched_hotels"] = matched.sort_values(by="Total_Score", ascending=False)
            
            st.session_state["show_limit"] = 20
            st.rerun()

        if st.session_state["has_searched"]:
            st.markdown(f"### {st.session_state.get('search_title', '')}")

            if st.session_state["matched_hotels"] is not None and not st.session_state["matched_hotels"].empty:
                all_matched_df = st.session_state["matched_hotels"]
                current_limit = st.session_state.get("show_limit", 20)
                display_df = all_matched_df.head(current_limit)

                for idx, row in display_df.iterrows():
                    with st.container(border=True):
                        chk_col, info_col = st.columns([0.05, 0.95])
                        with info_col:
                            st.markdown(f"## 🏨 {row['Hotel_Name']} — ⭐ Điểm: {format_number(row['Total_Score'], decimals=2)}")
                            st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                            
                            short_desc = get_short_description(row.get('Hotel_Description', 'Đang cập nhật'))
                            st.caption(f"📝 Mô tả: {short_desc}")
                            
                            with st.expander(" Xem toàn bộ mô tả"):
                                render_star_ratings(row)
                                st.write(str(row.get('Hotel_Description', 'Đang cập nhật')).strip())

                        with chk_col:
                            st.markdown("")
                            st.markdown("")
                            st.checkbox("", key=f"chk_{row['Hotel_ID']}")

                st.divider()

                if current_limit < len(all_matched_df):
                    if st.button("⬇️ Các lựa chọn khác (Tải thêm 20)"):
                        st.session_state["show_limit"] = current_limit + 20
                        st.rerun()

                with st.container(key="floating_config_container"):
                    with st.expander("⚙️ Cấu hình gợi ý cá nhân hóa (Số lượng, đặc điểm, vị trí)", expanded=False):
                        feature_input = st.text_input(
                            "Nhập đặc điểm / tiện ích khách sạn mong muốn (Tùy chọn):",
                            placeholder="Ví dụ: ho boi, bien, gym...",
                            key="float_feature_input"
                        )
                        num_personalized_recs = st.number_input(
                            "🎯 Số lượng gợi ý cá nhân hóa:",
                            min_value=1, max_value=100, value=10, step=1,
                            key="float_num_recs"
                        )
                        
                        if street_input.strip():
                            strict_street_cb = st.checkbox(
                                "📍 Bắt buộc nằm trên cùng đường",
                                key="float_strict_street_cb"
                            )
                            st.session_state["is_strict_street"] = strict_street_cb
                        else:
                            st.session_state["is_strict_street"] = False

                if st.button("🚀 OK - Nhận gợi ý cá nhân hóa", type="primary"):
                    selected_rows = []
                    for _, row in all_matched_df.iterrows():
                        if st.session_state.get(f"chk_{row['Hotel_ID']}", False):
                            selected_rows.append(row)

                    if selected_rows:
                        with st.spinner("Hệ thống đang phân tích sở thích cá nhân từ mô hình..."):
                            try:
                                chosen_df = pd.DataFrame(selected_rows)
                                selected_ids = chosen_df["Hotel_ID"].tolist()

                                hotel_indexer_model = st.session_state.hotel_indexer_model
                                als_hotel_model = st.session_state.als_hotel_model
                                labels = hotel_indexer_model.labels

                                id_to_idx = {str(label).strip(): i for i, label in enumerate(labels)}
                                chosen_indices = [id_to_idx[str(hid).strip()] for hid in selected_ids if str(hid).strip() in id_to_idx]

                                if not chosen_indices:
                                    st.warning("⚠️ Không thể mapping ID khách sạn đã chọn với mô hình.")
                                    st.stop()

                                item_factors_df = als_hotel_model.itemFactors.toPandas()
                                chosen_factors = item_factors_df[item_factors_df["id"].isin(chosen_indices)]
                                
                                if chosen_factors.empty:
                                    st.warning("⚠️ Không tìm thấy vector đặc trưng của các khách sạn đã chọn.")
                                    st.stop()

                                chosen_vectors = np.array(chosen_factors["features"].tolist())
                                user_features_vector = np.mean(chosen_vectors, axis=0)

                                all_vectors = np.array(item_factors_df["features"].tolist())
                                predicted_scores = np.dot(all_vectors, user_features_vector)
                                item_factors_df["predictedScore"] = predicted_scores
                                
                                recs_pd = item_factors_df.sort_values(by="predictedScore", ascending=False).head(350)

                                recs_pd["Hotel_ID"] = recs_pd["id"].apply(
                                    lambda idx: str(labels[idx]).strip() if idx < len(labels) else None
                                )
                                recs_pd["Hotel_ID"] = recs_pd["Hotel_ID"].astype(str).str.strip()
                                
                                cols_to_merge = [
                                    "Hotel_ID", "Hotel_Name", "Hotel_Address", "Hotel_Description", 
                                    "Address_norm", "Description_norm", "Location", "Cleanliness", 
                                    "Service", "Services", "Facilities", "Value_for_money", 
                                    "Comfort_and_room_quality", "comments_count", "Total_Score"
                                ]
                                existing_cols = [c for c in cols_to_merge if c in hotel_info.columns]

                                recs_expanded = recs_pd.merge(
                                     hotel_info[existing_cols].drop_duplicates(subset=["Hotel_ID"]),
                                     on="Hotel_ID", how="inner"  
                                )

                                recs_expanded = recs_expanded[~recs_expanded["Hotel_ID"].isin(selected_ids)]

                                if feature_input.strip():
                                    feature_keywords = [normalize_text(k) for k in feature_input.replace(',', ' ').split() if normalize_text(k)]
                                    if feature_keywords:
                                        def check_all_features(desc_norm):
                                            desc_str = str(desc_norm).lower()
                                            return all(kw in desc_str for kw in feature_keywords)

                                        feat_cond = recs_expanded["Description_norm"].apply(check_all_features)
                                        recs_expanded = recs_expanded[feat_cond]

                                if st.session_state.get("is_strict_street"):
                                    street_norm_target = st.session_state.get("search_keywords", "")
                                    saved_city_norm = st.session_state.get("city_norm_state", "")
                                    
                                    if street_norm_target:
                                        def check_street_recs(addr_norm, street_kw, c_norm):
                                            addr_str = str(addr_norm).lower()
                                            street_area = addr_str.split(c_norm)[0] if c_norm and (c_norm in addr_str) else addr_str
                                            return street_kw in street_area

                                        strict_cond = recs_expanded["Address_norm"].apply(
                                            lambda addr: check_street_recs(addr, street_norm_target, saved_city_norm)
                                        )
                                        recs_expanded = recs_expanded[strict_cond]

                                recs_expanded = recs_expanded.sort_values(by="predictedScore", ascending=False).head(int(num_personalized_recs))

                                st.session_state["recs_result_data"] = recs_expanded
                                st.session_state["chosen_hotels_data"] = chosen_df
                                st.session_state["show_recommendations"] = True
                                st.session_state["just_triggered"] = True
                                st.session_state["menu_selection_key"] = "✨ Personalized Results"
                                st.rerun()

                            except Exception as e:
                                error_details = traceback.format_exc()
                                st.error("❌ Lỗi hệ thống trong quá trình tính toán:")
                                st.code(error_details, language="python")
                    else:
                        st.warning("Vui lòng tích chọn ít nhất 1 khách sạn để hệ thống nhận diện gu của bạn.")

# =========================================================
# MỤC KẾT QUẢ GỢI Ý CÁ NHÂN HÓA
# =========================================================
elif menu_selection == "✨ Personalized Results":
    render_section_title("Personalized Results", "✨")
    render_hero_banner(
        title="Kết quả gợi ý cá nhân hóa",
        subtitle="Hệ thống đề xuất tự động dựa trên mô hình Collaborative Filtering dành riêng cho bạn.",
        cover_uri=personalized_cover_uri,
        bg_position="center 75%"
    )
    
    if st.session_state.get("just_triggered", False):
        st.balloons()
        st.session_state["just_triggered"] = False

    if st.session_state["recs_result_data"] is not None:
        recs_df = st.session_state["recs_result_data"]
        chosen_df = st.session_state["chosen_hotels_data"]
        
        st.markdown(f"### 🏆 Top {len(recs_df)} khách sạn mô hình gợi ý thêm dành riêng cho bạn:")

        if not recs_df.empty:
            for _, row in recs_df.iterrows():
                with st.container(border=True):
                    pred_score_fmt = format_number(row['predictedScore'], decimals=2)
                    st.markdown(f"## 🏆 {row['Hotel_Name']} — ✨ Độ tương thích dự đoán: `{pred_score_fmt}`")
                    st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                    
                    short_desc = get_short_description(row.get('Hotel_Description', 'Đang cập nhật'))
                    st.caption(f"📝 Mô tả đặc trưng: {short_desc}")
                    
                    with st.expander(" Xem toàn bộ mô tả"):
                        render_star_ratings(row)
                        st.write(str(row.get('Hotel_Description', 'Đang cập nhật')).strip())
        else:
            st.warning("Không tìm thấy đề xuất bổ sung nào đáp ứng đủ các tiêu chuẩn lọc nâng cao.")
            
        st.divider()
        
        st.markdown("### 📌 Các khách sạn bạn đã chọn làm input đầu vào:")
        if chosen_df is not None and not chosen_df.empty:
            for _, row in chosen_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"## 🏨 {row['Hotel_Name']} — ⭐ Điểm đánh giá: {format_number(row['Total_Score'], decimals=2)}")
                    st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                    
                    short_desc = get_short_description(row.get('Hotel_Description', 'Đang cập nhật'))
                    st.caption(f"📝 Mô tả: {short_desc}")
                    
                    with st.expander(" Xem toàn bộ mô tả"):
                        render_star_ratings(row)
                        st.write(str(row.get('Hotel_Description', 'Đang cập nhật')).strip())

        st.divider()
        if st.button("🔄 Thực hiện lượt tìm kiếm & gợi ý mới"):
            st.session_state["show_recommendations"] = False
            st.session_state["recs_result_data"] = None
            st.session_state["chosen_hotels_data"] = None
            st.session_state["menu_selection_key"] = "🎯 Gợi ý khách sạn"
            st.rerun()

# =========================================================
# MỤC KHÁM PHÁ THÔNG TIN KHÁCH SẠN
# =========================================================
elif menu_selection == "🏨 Khám phá thông tin khách sạn":
    render_section_title("Khám phá thông tin khách sạn", "🏨")
    render_hero_banner(
        title="Khám phá thông tin khách sạn",
        subtitle="Chọn một khách sạn để xem chi tiết về địa chỉ, điểm số, xếp hạng và mô tả đầy đủ.",
        cover_uri=explore_cover_uri,
        bg_position="center 75%"
    )

    if hotel_info is not None:
        hotel_names = sorted({str(name).strip() for name in hotel_info["Hotel_Name"].dropna() if str(name).strip()})
        selected_hotel_name = st.selectbox("Chọn khách sạn:", options=hotel_names)
        selected_hotel = hotel_info[hotel_info["Hotel_Name"].astype(str).str.strip() == str(selected_hotel_name).strip()].iloc[0]

        with st.container(border=True):
            st.markdown(f"## 🏨 {selected_hotel.get('Hotel_Name', 'Đang cập nhật')}")
            st.markdown(f"**📍 Địa chỉ:** {selected_hotel.get('Hotel_Address', 'Đang cập nhật')}")
            st.markdown(f"- ⭐ Total Score: {format_number(selected_hotel.get('Total_Score', 'N/A'), decimals=2)}")

            with st.expander("ℹ️ Thông tin chi tiết", expanded=True):
                render_star_ratings(selected_hotel)
                st.write(str(selected_hotel.get('Hotel_Description', 'Đang cập nhật')).strip())

# =========================================================
# MỤC CONTENT-BASED SIMILARITY
# =========================================================
elif menu_selection == "🧠 Tìm khách sạn tương đồng":
    render_section_title("Tìm khách sạn tương đồng", "🔎")
    render_hero_banner(
        title="Tìm khách sạn tương đồng",
        subtitle="Chọn một khách sạn để hệ thống gợi ý những khách sạn có nội dung tương đồng nhất bằng mô hình cosine similarity.",
        cover_uri=content_based_cover_data_uri
    )

    if hotel_info is not None and cosine_model is not None:
        hotel_names = sorted({str(name).strip() for name in hotel_info["Hotel_Name"].dropna() if str(name).strip()})
        selected_hotel_name = st.selectbox("Chọn khách sạn làm nền tảng:", options=hotel_names)
        count_input = st.text_input("Nhập số khách sạn tương đồng:", value="5", placeholder="Ví dụ: 5")

        if st.button("🔍 Tìm khách sạn tương đồng"):
            if str(count_input).strip() == "":
                st.error("Vui lòng nhập số vào ô khách sạn tương đồng.")
            else:
                try:
                    recommended_count = int(str(count_input).strip())
                except ValueError:
                    st.error("Vui lòng nhập số vào ô khách sạn tương đồng.")
                else:
                    if recommended_count <= 0:
                        st.error("Vui lòng nhập số vào ô khách sạn tương đồng.")
                    else:
                        data_search = cosine_model.get("data_search")
                        combined_text = cosine_model.get("combined_text")
                        vectorizer = cosine_model.get("vectorizer")
                        tfidf_matrix = cosine_model.get("tfidf_matrix")

                        if data_search is None or combined_text is None or vectorizer is None or tfidf_matrix is None:
                            st.error("Mô hình cosine similarity chưa sẵn sàng để chạy.")
                        else:
                            data_search = data_search.copy()
                            data_search["Hotel_Name_norm"] = data_search["Hotel_Name"].fillna("").astype(str).str.strip().str.lower()

                            selected_norm = str(selected_hotel_name).strip().lower()
                            matching_rows = data_search[data_search["Hotel_Name_norm"] == selected_norm]

                            if matching_rows.empty:
                                st.warning("Không tìm thấy khách sạn đã chọn trong mô hình cosine similarity.")
                            else:
                                selected_idx = int(matching_rows.index[0])
                                selected_text = combined_text[selected_idx] if selected_idx < len(combined_text) else ""

                                if not selected_text:
                                    st.warning("Không có nội dung mô tả để tính độ tương đồng cho khách sạn này.")
                                else:
                                    selected_vector = vectorizer.transform([selected_text])
                                    similarity_scores = cosine_similarity(selected_vector, tfidf_matrix).flatten()
                                    ranked_indices = np.argsort(similarity_scores)[::-1]
                                    ranked_indices = [idx for idx in ranked_indices if idx != selected_idx]

                                    top_indices = ranked_indices[:recommended_count]
                                    recs = data_search.iloc[top_indices].copy()
                                    recs["similarity_score"] = [float(similarity_scores[idx]) for idx in top_indices]
                                    recs = recs.sort_values(by="similarity_score", ascending=False)

                                    display_df = recs.merge(
                                        hotel_info[["Hotel_ID", "Hotel_Name", "Hotel_Address", "Hotel_Rank", "Total_Score", "Hotel_Description", "Cleanliness", "Service", "Comfort_and_room_quality", "Value_for_money"]].copy(),
                                        on="Hotel_Name",
                                        how="left",
                                        suffixes=("", "_info")
                                    )

                                    base_hotel = hotel_info[hotel_info["Hotel_Name"].astype(str).str.strip() == str(selected_hotel_name).strip()].iloc[0]

                                    st.subheader("1. Khách sạn được chọn làm nền tảng")
                                    with st.container(border=True):
                                        st.markdown(f"## 🏨 {base_hotel['Hotel_Name']}")
                                        st.markdown(f"**📍 Địa chỉ:** {base_hotel.get('Hotel_Address', 'Đang cập nhật')}")
                                        st.markdown(f"- ⭐ Total Score: {format_number(base_hotel.get('Total_Score', 'N/A'), decimals=2)}")
                                        st.markdown(f"- 🏅 Hotel Rank: {format_number(base_hotel.get('Hotel_Rank', 'N/A'), decimals=None)}")
                                        with st.expander("ℹ️ Thông tin chi tiết"):
                                            render_compact_hotel_details(base_hotel)

                                    st.subheader("2. Khách sạn đề xuất")
                                    if display_df.empty:
                                        st.info("Không tìm thấy khách sạn tương đồng phù hợp.")
                                    else:
                                        extra_recommendations = {}
                                        for _, row in display_df.iterrows():
                                            row_id = str(row.get("Hotel_ID", row.get("Hotel_Name", "")))
                                            extra_recs = get_similarity_recommendations(
                                                selected_hotel_name=row.get("Hotel_Name"),
                                                recommended_count=3,
                                                exclude_names=[row.get("Hotel_Name")]
                                            )
                                            extra_recommendations[row_id] = extra_recs

                                        for _, row in display_df.iterrows():
                                            row_id = str(row.get("Hotel_ID", row.get("Hotel_Name", "")))
                                            extra_result = extra_recommendations.get(row_id)

                                            with st.container(border=True):
                                                st.markdown(f"## 🏨 {row.get('Hotel_Name', 'Đang cập nhật')}")
                                                st.markdown(f"**📍 Địa chỉ:** {row.get('Hotel_Address', 'Đang cập nhật')}")
                                                st.markdown(f"- ⭐ Total Score: {format_number(row.get('Total_Score', 'N/A'), decimals=2)}")
                                                st.markdown(f"- 🏅 Hotel Rank: {format_number(row.get('Hotel_Rank', 'N/A'), decimals=None)}")
                                                with st.expander("ℹ️ Thông tin chi tiết"):
                                                    render_compact_hotel_details(row)

                                                with st.expander("📚 Xem thêm", expanded=False):
                                                    if extra_result is not None and not extra_result.empty:
                                                        st.markdown("**Một số khách sạn tương tự:**")
                                                        for _, extra_row in extra_result.iterrows():
                                                            with st.container(border=True):
                                                                st.markdown(f"## 🏨 {extra_row.get('Hotel_Name', 'Đang cập nhật')}")
                                                                st.markdown(f"**📍 Địa chỉ:** {extra_row.get('Hotel_Address', 'Đang cập nhật')}")
                                                                st.markdown(f"- ⭐ Total Score: {format_number(extra_row.get('Total_Score', 'N/A'), decimals=2)}")
                                                                st.markdown(f"- 🏅 Hotel Rank: {format_number(extra_row.get('Hotel_Rank', 'N/A'), decimals=None)}")
                                                                render_points_only_hotel_details(extra_row)
                                                    else:
                                                        st.info("Không có gợi ý bổ sung cho khách sạn này.")

                                                st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)

    else:
        st.warning("Không thể tải dữ liệu khách sạn hoặc mô hình cosine similarity.")

# =========================================================
# MỤC BẢNG PHÂN CÔNG VIỆC
# =========================================================
elif menu_selection == "👥 Bảng phân công việc":
    st.header("Bảng phân công công việc")
    st.markdown("""
    **Project 1: Recommender System**

    | Công việc | Người thực hiện | Email |
    |---|---|---|
    | Content-based filtering + GUI (Menu 1,2,5) | Huỳnh Bửu Khang | khang010504@gmail.com |
    | Collaborative filtering + GUI (Menu 3, 5) | Phạm Tuấn Kiệt | kiet40172@gmail.com |
    | Business Insights + GUI (Menu 4) | Nguyễn Hữu Nguyên Khôi | nguyenkhoinguyenhuu@gmail.com |
    """)
    st.divider()
    st.success("🎉 Cảm ơn các bạn đã sử dụng hệ thống của chúng tôi!")