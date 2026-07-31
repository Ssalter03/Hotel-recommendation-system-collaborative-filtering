import streamlit as st
import pandas as pd
import unicodedata
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexerModel

# =========================================================
# CẤU HÌNH TRANG WEB CHÍNH & TIÊU ĐỀ
# =========================================================
st.set_page_config(
    page_title="Hệ Thống Gợi Ý Khách Sạn Agoda",
    page_icon="🏨",
    layout="wide"
)

# Thêm CSS để ghim nút "OK - Nhận gợi ý" lơ lửng góc dưới bên phải (Floating Button)
st.markdown("""
<style>
    /* Nhắm mục tiêu vào nút có type="primary" để làm nút nổi */
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

# ===== Khởi tạo SparkSession và load model =====
if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder.appName("HotelRecommendation").getOrCreate()
if "hotel_indexer_model" not in st.session_state:
    try:
        st.session_state.hotel_indexer_model = StringIndexerModel.load("models/hotel_indexer")
    except Exception as e:
        st.error(f"Không thể load mô hình hotel_indexer: {e}")

# Các biến trạng thái Session State quản lý tương tác người dùng
if "show_recommendations" not in st.session_state: st.session_state["show_recommendations"] = False
if "recs_result_data" not in st.session_state: st.session_state["recs_result_data"] = None
if "chosen_hotels_data" not in st.session_state: st.session_state["chosen_hotels_data"] = None
# Quản lý tìm kiếm và phân trang (Load more)
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
# MỤC 1 & 2: KINH DOANH VÀ BÁO CÁO (GIỮ NGUYÊN)
# =========================================================
if menu_selection == "📋 Business Problem":
    st.title("💼 Bài toán Kinh Doanh (Business Problem)")
    st.write("Nền tảng đặt phòng trực tuyến như Agoda sở hữu lượng dữ liệu đánh giá khổng lồ...")

elif menu_selection == "📊 Evaluation & Report":
    st.title("📉 Đánh Giá Mô Hình & Báo Cáo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Thuật toán", "ALS (Matrix Factorization)")
    col2.metric("Rank", "6")
    col3.metric("Max Iterations", "10")

# =========================================================
# MỤC 3: RECOMMENDATION SYSTEM (ĐÃ NÂNG CẤP)
# =========================================================
elif menu_selection == "🎯 Recommendation System":
    st.title("🏨 Hotel Recommendation System")
    st.write("Tìm kiếm địa điểm lưu trú lý tưởng và đánh dấu các lựa chọn bạn yêu thích để hệ thống AI phân tích.")

    if hotel_info is not None:
        # ===== Khu vực Input & Filters =====
        street_input = st.text_input("Nhập tên đường (có hoặc không dấu):", placeholder="Ví dụ: Nguyen Trai, Tran Hung Dao...")
        strict_street_cb = st.checkbox("📍 Nơi nghỉ ngơi cá nhân hóa bắt buộc nằm trên cùng đường này")

        # Nút Tìm kiếm
        if st.button("🔍 Tìm khách sạn"):
            if street_input:
                street_norm = normalize_text(street_input)
                keywords = street_norm.split()
                
                if keywords:
                    # Lọc khách sạn theo từ khóa
                    condition = hotel_info["Address_norm"].apply(lambda addr: all(kw in str(addr) for kw in keywords))
                    matched = hotel_info[condition].copy()
                else:
                    matched = pd.DataFrame()

                if not matched.empty:
                    if "Total_Score" in matched.columns:
                        matched["Total_Score"] = matched["Total_Score"].fillna(0.0)
                        # Sắp xếp và lưu vào session_state toàn bộ kết quả tìm được
                        st.session_state["matched_hotels"] = matched.sort_values(by="Total_Score", ascending=False)
                        st.session_state["show_limit"] = 20 # Reset lại limit khi tìm mới
                        st.session_state["search_keywords"] = keywords
                        st.session_state["is_strict_street"] = strict_street_cb
                        st.toast(f"Tìm thấy tổng cộng {len(matched)} khách sạn.")
                else:
                    st.error("Không tìm thấy khách sạn nào trên cung đường này. Vui lòng thử từ khóa khác.")
                    st.session_state["matched_hotels"] = None

        # ===== Khu vực hiển thị danh sách & Load More =====
        if st.session_state["matched_hotels"] is not None:
            st.markdown("### 📌 Danh sách khách sạn phù hợp (Tick chọn khách sạn bạn thích):")
            
            all_matched_df = st.session_state["matched_hotels"]
            current_limit = st.session_state["show_limit"]
            display_df = all_matched_df.head(current_limit)

            # Lặp để render giao diện
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
                        # Tự động lưu trạng thái checkbox dựa trên key
                        st.checkbox("", key=f"chk_{row['Hotel_ID']}")

            st.divider()

            # Logic Load More
            total_matched = len(all_matched_df)
            if current_limit < total_matched:
                if st.button("⬇️ Các lựa chọn khác (Tải thêm 20)"):
                    st.session_state["show_limit"] += 20
                    st.rerun()
            else:
                st.info("🏁 Đã hết khách sạn trên đường này.")

            # ===== NÚT NỔI "OK" ĐỂ KÍCH HOẠT ALS =====
            # Nút này được đặt type="primary" để CSS ghim nó góc dưới màn hình
            if st.button("🚀 OK - Nhận gợi ý cá nhân hóa", type="primary"):
                # Thu thập các khách sạn người dùng đã check thông qua widget key
                selected_rows = []
                for _, row in all_matched_df.iterrows():
                    if st.session_state.get(f"chk_{row['Hotel_ID']}", False):
                        selected_rows.append(row)

                if selected_rows:
                    with st.spinner("AI đang học tập sở thích cá nhân của bạn..."):
                        chosen_df = pd.DataFrame(selected_rows)
                        selected_ids = chosen_df["Hotel_ID"].tolist()

                        spark = st.session_state.spark
                        hotel_indexer_model = st.session_state.hotel_indexer_model
                        labels = hotel_indexer_model.labels

                        user_index = 9999
                        user_pref_raw = spark.createDataFrame(
                            [(user_index, hid, 10.0) for hid in selected_ids],
                            ["userIndex", "Hotel ID", "Score"]
                        )
                        user_pref = hotel_indexer_model.transform(user_pref_raw).select("userIndex", "hotelIndex", "Score")

                        try:
                            reviews = spark.read.parquet("data/reviews.parquet")
                            reviews_small = reviews.select("userIndex", "hotelIndex", "Score")
                            reviews_with_new = reviews_small.union(user_pref)

                            als = ALS(maxIter=5, regParam=0.1, rank=6, userCol="userIndex", itemCol="hotelIndex", ratingCol="Score", coldStartStrategy="drop")
                            als_model_new = als.fit(reviews_with_new)
                            
                            recs = als_model_new.recommendForUserSubset(user_pref.select("userIndex").distinct(), 200) # Lấy dư ra để dự phòng cho strict filter
                            recs_pd = recs.toPandas()

                            recs_expanded = recs_pd.explode("recommendations")
                            recs_expanded["hotelIndex"] = recs_expanded["recommendations"].apply(lambda x: x[0])
                            recs_expanded["predictedScore"] = recs_expanded["recommendations"].apply(lambda x: x[1])

                            recs_expanded["Hotel_ID"] = recs_expanded["hotelIndex"].apply(
                                lambda idx: str(labels[idx]).strip() if idx < len(labels) else None
                            )

                            recs_expanded["Hotel_ID"] = recs_expanded["Hotel_ID"].astype(str).str.strip()
                            
                            recs_expanded = recs_expanded.merge(
                                hotel_info[["Hotel_ID", "Hotel_Name", "Hotel_Address", "Hotel_Description", "Address_norm"]].drop_duplicates(),
                                on="Hotel_ID", how="inner"
                            )

                            # Bỏ các khách sạn đã chọn
                            recs_expanded = recs_expanded[~recs_expanded["Hotel_ID"].isin(selected_ids)]

                            # ===== TÍNH NĂNG MỚI: BỘ LỌC ĐƯỜNG NGHIÊM NGẶT (STRICT FILTER) =====
                            if st.session_state["is_strict_street"]:
                                saved_keywords = st.session_state["search_keywords"]
                                if saved_keywords:
                                    strict_cond = recs_expanded["Address_norm"].apply(lambda addr: all(kw in str(addr) for kw in saved_keywords))
                                    recs_expanded = recs_expanded[strict_cond]

                            # Sắp xếp và lấy top 20 cuối cùng
                            recs_expanded = recs_expanded.sort_values(by="predictedScore", ascending=False).head(20)

                            # Cập nhật State để chuyển trang
                            st.session_state["recs_result_data"] = recs_expanded
                            st.session_state["chosen_hotels_data"] = chosen_df
                            st.session_state["show_recommendations"] = True
                            st.session_state["just_triggered"] = True
                            
                            st.success("Xử lý thành công! Đang chuyển hướng...")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Lỗi hệ thống trong quá trình retrain ALS: {e}")
                else:
                    st.warning("Vui lòng tick chọn ít nhất 1 khách sạn trên giao diện.")

# =========================================================
# MỤC KẾT QUẢ GỢI Ý CÁ NHÂN HÓA 
# =========================================================
elif menu_selection == "✨ Personalized Results":
    st.title("🎯 Kết Quả Gợi Ý Cá Nhân Hóa Dành Riêng Cho Bạn")
    st.subheader("Dựa trên thuật toán Collaborative Filtering mô phỏng thời gian thực")

    if st.session_state["recs_result_data"] is not None:
        recs_df = st.session_state["recs_result_data"]
        chosen_df = st.session_state["chosen_hotels_data"]
        
        # 1. HIỂN THỊ CÁC KHÁCH SẠN DO MODEL ĐỀ XUẤT
        st.markdown("### 🏆 Danh sách khách sạn mô hình gợi ý thêm:")
        
        # Báo cáo trạng thái bộ lọc
        if st.session_state["is_strict_street"]:
            st.info("🔒 Tính năng *'Cá nhân hóa bắt buộc nằm trên cùng đường'* ĐANG ĐƯỢC BẬT. Danh sách dưới đây chỉ gồm các khách sạn nằm trong tuyến đường bạn yêu cầu.")

        if not recs_df.empty:
            for _, row in recs_df.iterrows():
                with st.container(border=True):
                    pred_score_fmt = round(float(row['predictedScore']), 2)
                    st.markdown(f"## 🏆 {row['Hotel_Name']} — ✨ Độ tương thích AI: `{pred_score_fmt}`")
                    st.markdown(f"**📍 Địa chỉ:** *{row['Hotel_Address']}*")
                    st.caption(f"📝 Mô tả đặc trưng: {row.get('Hotel_Description', 'Đang cập nhật')}")
        else:
            if st.session_state["is_strict_street"]:
                st.warning("Rất tiếc! Mô hình không tìm thấy khách sạn nào khác CÙNG TRÊN ĐƯỜNG NÀY phù hợp với sở thích của bạn. Bạn có thể tắt tính năng lọc đường để nhận gợi ý đa dạng hơn.")
            else:
                st.warning("Mô hình không tìm thấy đề xuất bổ sung nào khác biệt bên ngoài danh sách lựa chọn của bạn.")
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.divider()
        
        # 2. HIỂN THỊ MỤC CÁC KHÁCH SẠN USER ĐÃ CHỌN Ở DƯỚI CÙNG
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Học viên 1")
        st.markdown(
            """
            - **Họ và tên:** [Họ tên HV 1 của nhóm bạn]
            - **Email:** hv1_email@gmail.com
            - **Mã học viên:** HV001
            """
        )
        with st.container(border=True):
            st.markdown("**Phân công nhiệm vụ:**")
            st.write("- Thu thập, cào và làm sạch dữ liệu khách sạn từ nền tảng Agoda.")
            st.write("- Phân tích và xây dựng mô hình Collaborative Filtering PySpark ALS trên Google Colab.")

    with col2:
        st.subheader("👤 Học viên 2")
        st.markdown(
            """
            - **Họ và tên:** [Họ tên HV 2 của nhóm bạn]
            - **Email:** hv2_email@gmail.com
            - **Mã học viên:** HV002
            """
        )
        with st.container(border=True):
            st.markdown("**Phân công nhiệm vụ:**")
            st.write("- Đóng gói, xuất các file mô hình indexer và file dữ liệu Parquet.")
            st.write("- Thiết kế lập trình ứng dụng GUI Web Streamlit tương tác thời gian thực trên VS Code.")
            
    st.divider()
    st.success("🎉 Dự án được hoàn thiện, phân vùng thư mục chuẩn và đáp ứng đầy đủ yêu cầu nghiệp vụ.")