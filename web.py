import streamlit as st
import pandas as pd
import unicodedata
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql import functions as F
import traceback

st.markdown("""
    <style>
        /* Tăng size chữ chung cho toàn bộ text của Streamlit */
        .stText, p, span {
            font-size: 20px !important;
        }
        /* Tăng size cho các hàng text nhỏ (st.caption) */
        .stCaptionText, caption {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)


# =========================================================
# CẤU HÌNH TRANG WEB CHÍNH & TIÊU ĐỀ
# =========================================================
st.set_page_config(
    page_title="Hệ Thống Gợi Ý Khách Sạn Sử Dụng Model ALS Collaborative Filtering",
    page_icon="🏨",
    layout="wide"
)

# Thêm CSS để ghim nút "OK - Nhận gợi ý" lơ lửng góc dưới bên phải (Floating Button)
st.markdown("""
<style>
    button[kind="primary"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        z-index: 9999 !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5) !important;
        border-radius: 30px !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        transition: 0.3s;
    }
    button[kind="primary"]:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ===== Hàm chuẩn hóa text =====
def normalize_text(s):
    if pd.isna(s):
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

# ===== Load dữ liệu =====
@st.cache_data
def load_base_data():
    try:
        hotel_info = pd.read_csv("data/hotel_info.csv")
        hotel_info["Address_norm"] = hotel_info["Hotel_Address"].apply(normalize_text)
        hotel_info["Hotel_ID"] = hotel_info["Hotel_ID"].astype(str).str.strip()
        return hotel_info
    except Exception as e:
        st.error(f"Lỗi khi đọc file hotel_info.csv: {e}")
        return None

hotel_info = load_base_data()

# ===== Tự động trích xuất danh sách tỉnh/thành phố phổ biến tại VN từ data =====
def get_vietnam_cities(df):
    if df is None or df.empty:
        return ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Nha Trang", "Vũng Tàu", "Đà Lạt", "Phú Quốc", "Hội An"]
    
    popular_cities = {
        "Hồ Chí Minh": ["ho chi minh", "hcm", "sài gòn", "sai gon", "distric"],
        "Hà Nội": ["ha noi", "hanoi"],
        "Đà Nẵng": ["da nang", "danang"],
        "Nha Trang": ["nha trang", "nhatrang"],
        "Vũng Tàu": ["vung tau", "vungtau"],
        "Đà Lạt": ["da lat", "dalat"],
        "Phú Quốc": ["phu quoc", "phuquoc"],
        "Hội An": ["hoi an", "hoian"],
        "Huế": ["hue"],
        "Hạ Long": ["ha long", "halong"]
    }
    
    available_cities = []
    sample_addresses = df["Address_norm"].str.cat(sep=" ")
    for city_name, keywords in popular_cities.items():
        if any(kw in sample_addresses for kw in keywords):
            available_cities.append(city_name)
            
    return available_cities if available_cities else ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng"]

cities_list = get_vietnam_cities(hotel_info)

# ===== Khởi tạo SparkSession và load model =====
from pyspark.sql import SparkSession

if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("HotelRecsStreamlit") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "2") \
        .config("spark.broadcast.compress", "true") \
        .config("spark.rdd.compress", "true") \
        .getOrCreate()
        
if "hotel_indexer_model" not in st.session_state:
    try:
        st.session_state.hotel_indexer_model = StringIndexerModel.load("models/hotel_indexer")
    except Exception as e:
        st.error(f"Không thể load mô hình hotel_indexer: {e}")

if "als_hotel_model" not in st.session_state:
    try:
        # Tải mô hình ALS đã sửa đúng tên đường dẫn thư mục từ ảnh: models/als_hotel_model
        st.session_state.als_hotel_model = ALSModel.load("models/als_hotel_model")
    except Exception as e:
        st.error(f"Không thể load mô hình ALS đã train sẵn từ thư mục 'models/als_hotel_model': {e}")

# Quản lý các biến trạng thái Session State
if "show_recommendations" not in st.session_state: st.session_state["show_recommendations"] = False
if "recs_result_data" not in st.session_state: st.session_state["recs_result_data"] = None
if "chosen_hotels_data" not in st.session_state: st.session_state["chosen_hotels_data"] = None
if "show_limit" not in st.session_state: st.session_state["show_limit"] = 20
if "matched_hotels" not in st.session_state: st.session_state["matched_hotels"] = None
if "search_keywords" not in st.session_state: st.session_state["search_keywords"] = []
if "is_strict_street" not in st.session_state: st.session_state["is_strict_street"] = False

# =========================================================
# MENU ĐIỀU HƯỚNG BÊN TRÁI GUI
# =========================================================
st.sidebar.title("🧭 HỆ THỐNG MENU")

menu_options = [
    "📋 Business Problem", 
    "📊 Evaluation & Report", 
    "🎯 Recommendation System", 
    "👥 Member Information & Tasks"
]
if st.session_state["show_recommendations"]:
    menu_options.insert(3, "✨ Personalized Results")

menu_selection = st.sidebar.radio("Chọn chức năng:", menu_options)

if st.session_state["show_recommendations"] and menu_selection != "✨ Personalized Results" and st.session_state.get("just_triggered", False):
    st.session_state["just_triggered"] = False
    st.rerun()

# =========================================================
# MỤC 1 & 2: KINH DOANH VÀ BÁO CÁO
# =========================================================
if menu_selection == "📋 Business Problem":
    st.title("💼 Bài toán Kinh Doanh (Business Problem)")
    st.write("Hệ thống gợi ý khách sạn sử dụng thuật toán Collaborative Filtering (Lọc cộng tác) thông qua mô hình ALS (Alternating Least Squares) trên nền tảng Apache Spark được triển khai để giải quyết trực tiếp các nút thắt kinh doanh này:")
    st.write("- Giải quyết bài toán dữ liệu thưa thớt (Data Sparsity): Khách hàng thường chỉ tương tác (đặt phòng, đánh giá) với một số lượng rất nhỏ khách sạn trên hệ thống. ALS giúp dự đoán chính xác mức độ yêu thích của khách hàng đối với các khách sạn họ chưa từng biết tới dựa trên hành vi của các nhóm người dùng có sở thích tương đồng.")
    st.write("- Tối ưu hóa hiệu năng với dữ liệu lớn (Scalability): Với hàng triệu người dùng và hàng trăm nghìn khách sạn, các hệ thống thông thường sẽ bị nghẽn mạch. Khả năng tính toán phân tán của Spark giúp xử lý dữ liệu hành vi khổng lồ theo thời gian thực, đảm bảo gợi ý luôn mượt mà và cập nhật.")

elif menu_selection == "📊 Evaluation & Report":
    st.title("Đánh Giá Mô Hình & Báo Cáo")
    st.text("Thuật toán: ALS (Matrix Factorization)")
    st.write("") 
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Rank", value="6")
    col2.metric(label="Max Iterations", value="10")
    col3.metric(label="RMSE", value="0.189732")

# =========================================================
# MỤC 3: RECOMMENDATION SYSTEM (LOAD MODEL & ĐỒNG BỘ LOGIC)
# =========================================================
elif menu_selection == "🎯 Recommendation System":
    st.title("🏨 Hotel Recommendation System")
    st.write("Tìm kiếm địa điểm lưu trú lý tưởng theo Tỉnh/Thành phố hoặc Tuyến đường bạn yêu thích.")

    if hotel_info is not None:
        if "has_searched" not in st.session_state: st.session_state["has_searched"] = False
        if "matched_hotels" not in st.session_state: st.session_state["matched_hotels"] = None
        if "search_title" not in st.session_state: st.session_state["search_title"] = ""
        if "is_strict_street" not in st.session_state: st.session_state["is_strict_street"] = False
        if "search_keywords" not in st.session_state: st.session_state["search_keywords"] = ""
        if "city_norm_state" not in st.session_state: st.session_state["city_norm_state"] = ""

        # ===== Khoảng nhập liệu đầu vào =====
        selected_city = st.selectbox("🌆 Chọn Tỉnh / Thành phố:", options=cities_list)
        city_norm = normalize_text(selected_city).strip() 

        street_input = st.text_input(
            "Nhập tên đường (Tùy chọn - Có hoặc không dấu):", 
            placeholder="Ví dụ: Nguyen Trai, Le Loi... (Để trống để xem toàn bộ thành phố)"
        )

        if street_input.strip():
            strict_street_cb = st.checkbox(
                "📍 Nơi nghỉ ngơi cá nhân hóa bắt buộc nằm trên cùng đường này",
                key="strict_street_checkbox_state"
            )
            st.session_state["is_strict_street"] = strict_street_cb
        else:
            st.session_state["is_strict_street"] = False
            st.caption("ℹ️ *Đang hiển thị chế độ toàn thành phố. Tính năng bắt buộc nằm trên cùng đường đã được ẩn.*")

        # NÚT BẤM TÌM KIẾM
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
                if street_input.strip():
                    street_norm = normalize_text(street_input).strip()
                    st.session_state["search_keywords"] = street_norm
                    
                    def check_street_smart_final(addr_norm, street_kw, c_norm):
                        addr_str = str(addr_norm).lower()
                        if c_norm in addr_str:
                            street_area = addr_str.split(c_norm)[0]
                        else:
                            street_area = addr_str
                        return street_kw in street_area

                    street_condition = city_hotels["Address_norm"].apply(
                        lambda addr: check_street_smart_final(addr, street_norm, city_norm)
                    )
                    matched = city_hotels[street_condition].copy()
                    
                    if matched.empty:
                        st.session_state["matched_hotels"] = None
                        st.session_state["search_title"] = f'📭 Không tìm thấy khách sạn nào nằm chính xác trên đường "{street_input.strip()}" tại {selected_city}.'
                    else:
                        st.session_state["search_title"] = f'📌 Danh sách khách sạn phù hợp nằm trên đường "{street_input.strip()}", {selected_city}:'
                        if "Total_Score" in matched.columns:
                            matched["Total_Score"] = matched["Total_Score"].fillna(0.0)
                        st.session_state["matched_hotels"] = matched.sort_values(by="Total_Score", ascending=False)
                else:
                    matched = city_hotels
                    st.session_state["search_keywords"] = ""
                    st.session_state["search_title"] = f"📌 Danh sách khách sạn phù hợp ở {selected_city}:"
                    
                    if "Total_Score" in matched.columns:
                        matched["Total_Score"] = matched["Total_Score"].fillna(0.0)
                    st.session_state["matched_hotels"] = matched.sort_values(by="Total_Score", ascending=False)
            
            st.session_state["show_limit"] = 20
            st.rerun()

        # ===== HIỂN THỊ KẾT QUẢ DANH SÁCH =====
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
                            st.markdown(f"## 🏨 {row['Hotel_Name']} — ⭐ Điểm: {row['Total_Score']}")
                            st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                            st.caption(f"📝 Mô tả: {row.get('Hotel_Description', 'Đang cập nhật')}")
                        with chk_col:
                            st.write("")
                            st.write("")
                            st.checkbox("", key=f"chk_{row['Hotel_ID']}")

                st.divider()

                # Nút Phân trang
                total_matched = len(all_matched_df)
                if current_limit < total_matched:
                    if st.button("⬇️ Các lựa chọn khác (Tải thêm 20)"):
                        st.session_state["show_limit"] = current_limit + 20
                        st.rerun()
                else:
                    st.info("🏁 Đã hiển thị hết danh sách khách sạn tìm thấy.")

                # ===== NÚT NỔI "OK - NHẬN GỢI Ý CÁ NHÂN HÓA" =====
                if st.button("🚀 OK - Nhận gợi ý cá nhân hóa", type="primary"):
                    selected_rows = []
                    for _, row in all_matched_df.iterrows():
                        if st.session_state.get(f"chk_{row['Hotel_ID']}", False):
                            selected_rows.append(row)

                    if selected_rows:
                        with st.spinner("Hệ thống đang phân tích sở thích cá nhân từ mô hình có sẵn..."):
                            try:
                                import numpy as np
                                chosen_df = pd.DataFrame(selected_rows)
                                selected_ids = chosen_df["Hotel_ID"].tolist()

                                hotel_indexer_model = st.session_state.hotel_indexer_model
                                als_hotel_model = st.session_state.als_hotel_model
                                labels = hotel_indexer_model.labels

                                # 1. Chuyển đổi ID khách sạn được chọn sang index tương ứng
                                # Dùng list comprehension để tìm index cho nhanh, tránh dùng Spark DataFrame
                                id_to_idx = {str(label).strip(): i for i, label in enumerate(labels)}
                                chosen_indices = [id_to_idx[str(hid).strip()] for hid in selected_ids if str(hid).strip() in id_to_idx]

                                if not chosen_indices:
                                    st.warning("⚠️ Không thể mapping các ID khách sạn đã chọn với mô hình học máy.")
                                    st.stop()

                                # 2. Lấy thông tin Item Factors (chuyển sang Pandas để tính bằng NumPy cho siêu nhẹ)
                                item_factors_df = als_hotel_model.itemFactors.toPandas()
                                
                                # Lọc lấy vector của các khách sạn người dùng đã chọn
                                chosen_factors = item_factors_df[item_factors_df["id"].isin(chosen_indices)]
                                
                                if chosen_factors.empty:
                                    st.warning("⚠️ Không tìm thấy vector đặc trưng của các khách sạn đã chọn.")
                                    st.stop()

                                # 3. Tính Vector trung bình đại diện cho Sở thích (User Profile Vector) bằng NumPy
                                chosen_vectors = np.array(chosen_factors["features"].tolist())
                                user_features_vector = np.mean(chosen_vectors, axis=0)

                                # 4. Tính toán khoảng cách Dot Product bằng ma trận NumPy (Tốc độ ánh sáng, không bị treo)
                                all_vectors = np.array(item_factors_df["features"].tolist())
                                predicted_scores = np.dot(all_vectors, user_features_vector)
                                
                                item_factors_df["predictedScore"] = predicted_scores
                                
                                # Lấy top các phần tử phù hợp nhất
                                recs_pd = item_factors_df.sort_values(by="predictedScore", ascending=False).head(350)

                                # 5. Bản đồ hóa ID thực tế và Merge thông tin khách sạn
                                recs_pd["Hotel_ID"] = recs_pd["id"].apply(
                                    lambda idx: str(labels[idx]).strip() if idx < len(labels) else None
                                )
                                recs_pd["Hotel_ID"] = recs_pd["Hotel_ID"].astype(str).str.strip()
                                
                                recs_expanded = recs_pd.merge(
                                     hotel_info[["Hotel_ID", "Hotel_Name", "Hotel_Address", "Hotel_Description", "Address_norm"]].drop_duplicates(),
                                     on="Hotel_ID", how="inner"  
                                )

                                # Loại bỏ các khách sạn người dùng đã chọn từ trước
                                recs_expanded = recs_expanded[~recs_expanded["Hotel_ID"].isin(selected_ids)]

                                # BỘ LỌC ĐƯỜNG NGHIÊM NGẶT
                                if st.session_state.get("is_strict_street"):
                                    street_norm_target = st.session_state.get("search_keywords", "")
                                    saved_city_norm = st.session_state.get("city_norm_state", "")
                                    
                                    if street_norm_target:
                                        def check_street_smart_recs_fixed(addr_norm, street_kw, c_norm):
                                            addr_str = str(addr_norm).lower()
                                            if c_norm and (c_norm in addr_str):
                                                street_area = addr_str.split(c_norm)[0]
                                            else:
                                                street_area = addr_str
                                            return street_kw in street_area

                                        strict_cond = recs_expanded["Address_norm"].apply(
                                            lambda addr: check_street_smart_recs_fixed(addr, street_norm_target, saved_city_norm)
                                        )
                                        recs_expanded = recs_expanded[strict_cond]

                                # Lấy top 20 kết quả phù hợp nhất
                                recs_expanded = recs_expanded.sort_values(by="predictedScore", ascending=False).head(20)

                                st.session_state["recs_result_data"] = recs_expanded
                                st.session_state["chosen_hotels_data"] = chosen_df
                                st.session_state["show_recommendations"] = True
                                st.session_state["just_triggered"] = True
                                
                                st.success("Xử lý thành công bằng phương pháp Vector Dot-Product cực nhanh!")
                                st.rerun()

                            except Exception as e:
                                error_details = traceback.format_exc()
                                st.error("❌ Lỗi hệ thống trong quá trình tính toán:")
                                st.code(error_details, language="python")
                    else:
                        st.warning("Vui lòng tick chọn ít nhất 1 khách sạn để hệ thống nhận diện được gu của bạn.")
                        
# =========================================================
# MỤC KẾT QUẢ GỢI Ý CÁ NHÂN HÓA 
# =========================================================
elif menu_selection == "✨ Personalized Results":
    st.title("🎯 Kết Quả Gợi Ý Cá Nhân Hóa Dành Riêng Cho Bạn")
    
    if st.session_state.get("just_triggered", False):
        st.balloons()
        st.session_state["just_triggered"] = False
        
    st.subheader("Hệ thống đề xuất tự động dựa trên mô hình Collaborative Filtering (Pre-trained)")

    if st.session_state["recs_result_data"] is not None:
        recs_df = st.session_state["recs_result_data"]
        chosen_df = st.session_state["chosen_hotels_data"]
        
        st.markdown("### 🏆 Danh sách khách sạn mô hình gợi ý thêm:")
        
        if st.session_state["is_strict_street"]:
            st.info("🔒 Bộ lọc nâng cao: *'Nơi nghỉ ngơi cá nhân hóa bắt buộc nằm trên cùng đường'* đang bật.")

        if not recs_df.empty:
            for _, row in recs_df.iterrows():
                with st.container(border=True):
                    pred_score_fmt = round(float(row['predictedScore']), 2)
                    st.markdown(f"## 🏆 {row['Hotel_Name']} — ✨ Độ tương thích dự đoán: `{pred_score_fmt}`")
                    st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                    st.caption(f"📝 Mô tả đặc trưng: {row.get('Hotel_Description', 'Đang cập nhật')}")
        else:
            st.warning("Không tìm thấy đề xuất bổ sung nào đáp ứng đủ tiêu chuẩn lọc hiện tại ngoài danh sách lựa chọn gốc.")
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.divider()
        
        # HIỂN THỊ CÁC KHÁCH SẠN USER ĐÃ CHỌN XUỐNG DƯỚI CÙNG
        st.markdown("### 📌 Hotels you chose (Danh sách khách sạn bạn đã chọn trước đó):")
        if chosen_df is not None and not chosen_df.empty:
            for _, row in chosen_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"## 🏨 {row['Hotel_Name']} — ⭐ Điểm đánh giá: {row['Total_Score']}")
                    st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                    st.caption(f"📝 Mô tả: {row.get('Hotel_Description', 'Đang cập nhật')}")

        st.divider()
        if st.button("🔄 Thực hiện lượt tìm kiếm & gợi ý mới"):
            st.session_state["show_recommendations"] = False
            st.session_state["recs_result_data"] = None
            st.session_state["chosen_hotels_data"] = None
            st.rerun()

# =========================================================
# MỤC 4: THÔNG TIN NHÓM & PHÂN CÔNG
# =========================================================
elif menu_selection == "👥 Member Information & Tasks":
    st.title("👥 Thông Tin Nhóm Thực Hiện & Phân Công Công Việc")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Học viên 1")
        st.markdown(
            """
            - **Họ và tên:** [Nguyen Huu Nguyen Khoi]
            - **Email:** nguyenkhoinguyenhuu@gmail.com
            """
        )
        with st.container(border=True):
            st.markdown("**Phân công nhiệm vụ:**")
            st.write("- Business Insights Project 1")
            st.write("- Data preprocessing (Minor) + Bài toán 2 (yêu cầu 4, 5) + GUI (Menu 4)")

    with col2:
        st.subheader("👤 Học viên 2")
        st.markdown(
            """
            - **Họ và tên:** [Pham Tuan Kiet]
            - **Email:** kiet40172@gmail.com
            """
        )
        with st.container(border=True):
            st.markdown("**Phân công nhiệm vụ:**")
            st.write("- Dùng ALSPySpark và ML truyền thống (RandomForest)để huấn luyện mô hình Collaborative Filtering gợi ý hotel")
            st.write("- Hoàn thành bài toán 1,2 và 3 ở project 2.")
            st.write("- Thiết kế lập trình ứng dụng GUI Web Streamlit: Hotel Recomendation system dùng model collaborative-filtering.")
                
    with col3:
        st.subheader("👤 Học viên 3")
        st.markdown(
            """
            - **Họ và tên:** [Huynh Buu Khang]
            - **Email:** khang010504@gmail.com
            """
        )
        with st.container(border=True):
            st.markdown("**Phân công nhiệm vụ:**")
            st.write("- Content-based filtering Project 1")
            st.write("- Data preprocessing (Main) + EDA + bài toán 1 + GUI (Menu 1, 2, 3)")
            
    st.divider()
    st.success("🎉 Cảm ơn các bạn đã sử dụng hệ thống của chúng tôi!")