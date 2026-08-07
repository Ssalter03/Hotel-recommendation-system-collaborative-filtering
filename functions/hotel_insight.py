import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def hotel_overview(df, hotel_name):
    """
    Trả về thông tin tổng quan của khách sạn.
    """

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Hotel not found.")

    hotel = hotel.iloc[0]

    overview = {
        "Hotel Name": hotel["Hotel_Name"],
        "Rank (Star)": hotel["Hotel_Rank"],
        "Address": hotel["Hotel_Address"],
        "Total Score": hotel["Total_Score"],
        "Comments": hotel["comments_count"],
    }

    return overview

#============================================
def review_reliability(df, hotel_name):

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Không tìm thấy khách sạn.")

    comments = int(hotel.iloc[0]["comments_count"])

    if comments < 10:
        level = "Thấp"
        message = (
            "⚠️ Khách sạn này hiện có khá ít đánh giá từ khách hàng. "
            "Điểm số có thể chưa phản ánh đầy đủ chất lượng thực tế."
        )

    elif comments < 50:
        level = "Trung bình"
        message = (
            "💬 Số lượng đánh giá ở mức vừa phải. "
            "Bạn nên tham khảo thêm các tiêu chí khác trước khi đưa ra quyết định."
        )

    else:
        level = "Cao"
        message = (
            "⭐ Điểm đánh giá được tổng hợp từ nhiều phản hồi của khách hàng, "
            "do đó có độ tin cậy khá tốt."
        )

    return {
        "comments": comments,
        "reliability": level,
        "message": message
    }
    
#============================================
def plot_hotel_radar(df, hotel_name, score_cols):
    """
    Radar chart so sánh khách sạn với trung bình toàn bộ dataset.
    """

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Hotel not found.")

    hotel = hotel.iloc[0]

    hotel_scores = hotel[score_cols].astype(float).values
    avg_scores = df[score_cols].mean().values

    labels = score_cols
    num_vars = len(labels)

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    hotel_scores = np.concatenate((hotel_scores, [hotel_scores[0]]))
    avg_scores = np.concatenate((avg_scores, [avg_scores[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, hotel_scores, linewidth=2, label="Hotel")
    ax.fill(angles, hotel_scores, alpha=0.25)

    ax.plot(angles, avg_scores, linewidth=2, label="Average Hệ thống")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
    labels,
    fontsize=12,   # độ lớn chữ
    )
    
    ax.tick_params(
    axis="x",
    pad=15
    )

    ax.set_ylim(5,10)

    plt.legend()
    
    return fig

def plot_score_difference(df, hotel_name, score_cols):

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Hotel not found.")

    hotel = hotel.iloc[0]


    hotel_scores = (
        hotel[score_cols]
        .fillna(df[score_cols].mean())
        .astype(float)
    )

    avg_scores = df[score_cols].mean()

    diff = hotel_scores - avg_scores


    fig, ax = plt.subplots(figsize=(8,4))



    # Range tự động
    x_min = diff.min() - 0.2
    x_max = diff.max() + 0.2

    # tránh trường hợp tất cả gần 0
    if x_min == x_max:
        x_min -= 0.2
        x_max += 0.2

    ax.set_xlim(
        x_min,
        x_max
    )

    # ==========================
    # Main bar
    colors = [
        "tab:blue" if x >= 0 else "tab:red"
        for x in diff
    ]


    ax.barh(
        score_cols,
        diff,
        height=0.4,
        color=colors,
        zorder=2
    )

    # ==========================
    # Đường chuẩn 0
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=3
    )

    # ==========================
    # Label
    for i, value in enumerate(diff):

        offset = 0.01

        if value >= 0:
            x_pos = value + offset
            ha = "left"
        else:
            x_pos = value - offset
            ha = "right"

        ax.text(
            x_pos,
            i,
            f"{value:+.2f}",
            va="center",
            ha=ha,
            fontsize=10
        )

    # ==========================
    # Axis
    ax.set_xlabel(
        f"Chênh lệch điểm của '{hotel_name}' và TB hệ thống"
    )

    # ==========================
    # Remove borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.4
    )

    plt.tight_layout()

    return fig
    

def total_score_percentile(df, hotel_name):
    """
    Trả về Total Score và percentile của khách sạn.
    """

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Hotel not found.")

    score = hotel.iloc[0]["Total_Score"]

    if pd.isna(score):
        return None, None

    percentile = (
        df["Total_Score"]
        .rank(pct=True)
        .loc[hotel.index[0]]
        * 100
    )

    return score, round(percentile, 1)
    
def total_score_insight(percentile):

    if percentile >= 90:
        level = "Xuất sắc"
        message = (
            "🏆 Khách sạn này thuộc nhóm 10% khách sạn có điểm đánh giá cao nhất. "
            "Đây là một lựa chọn nổi bật với trải nghiệm khách hàng rất tốt."
        )

    elif percentile >= 75:
        level = "Rất tốt"
        message = (
            "🌟 Khách sạn này nằm trong top 25% khách sạn được đánh giá tốt. "
            "Chất lượng tổng thể vượt trội so với phần lớn khách sạn khác."
        )

    elif percentile >= 50:
        level = "Tốt"
        message = (
            "😊 Khách sạn này có điểm đánh giá cao hơn mức trung bình. "
            "Đây là lựa chọn khá ổn với trải nghiệm tương đối tích cực."
        )

    elif percentile >= 25:
        level = "Trung bình"
        message = (
            "🤔 Khách sạn có mức đánh giá chưa thật sự nổi bật. "
            "Bạn nên xem thêm các tiêu chí chi tiết như vị trí, dịch vụ và tiện nghi."
        )

    else:
        level = "Cần cân nhắc"
        message = (
            "⚠️ Khách sạn này đang nằm trong nhóm có điểm đánh giá thấp hơn "
            "so với phần lớn khách sạn khác trong hệ thống."
        )

    return {
        "percentile": percentile,
        "level": level,
        "message": message
    }
    
    
def generate_hotel_report(
    df,
    hotel_name,
    score_cols
):

    overview = hotel_overview(
        df,
        hotel_name
    )


    reliability = review_reliability(
        df,
        hotel_name
    )


    score, percentile = total_score_percentile(
        df,
        hotel_name
    )


    insight = total_score_insight(
        percentile
    )


    radar = plot_hotel_radar(
        df,
        hotel_name,
        score_cols
    )


    score_diff = plot_score_difference(
        df,
        hotel_name,
        score_cols
    )


    return {
        "overview": overview,
        "reliability": reliability,
        "score": score,
        "percentile": percentile,
        "insight": insight,
        "radar": radar,
        "score_difference": score_diff
    }


def plot_customer_type_pie(df, hotel_name):
    """
    Vẽ pie chart phân bố Customer Type của một khách sạn cụ thể.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn được chọn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Đếm Customer Type
    customer_counts = (
        hotel_df["Customer_Type"]
        .value_counts()
        .reset_index()
    )

    customer_counts.columns = [
        "Customer_Type",
        "Count"
    ]


    # 🌍 Travel / World Map Theme
    customer_colors = {
        "International": "#264653",  # Deep ocean navy
        "Domestic": "#2A9D8F"        # Travel teal
    }


    fig = px.pie(
        customer_counts,

        names="Customer_Type",

        values="Count",

        hole=0.45,

        color="Customer_Type",

        color_discrete_map=customer_colors
    )


    # Style pie
    fig.update_traces(

        textposition="inside",

        textinfo="label+percent",

        textfont=dict(
            size=11,
            color="white",
            family="Arial"
        ),

        marker=dict(
            line=dict(
                color="white",
                width=2
            )
        ),

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Reviews: %{value}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        )
    )


    fig.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        legend_title="Customer Type",

        legend=dict(
            font=dict(
                size=11,
                color="#222222"
            )
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",


        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial"
        ),


        margin=dict(
            t=40,
            l=20,
            r=20,
            b=20
        ),

        height=500
    )


    return fig


def plot_nationality_pie(df, hotel_name):
    """
    Vẽ pie chart phân bố Nationality_Group của một khách sạn cụ thể,
    loại bỏ khách Việt Nam.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn được chọn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Loại bỏ khách Việt Nam
    foreign_df = hotel_df[
        hotel_df["Nationality_Group"] != "Việt Nam"
    ]


    if foreign_df.empty:
        raise ValueError(
            "Khách sạn này không có khách quốc tế."
        )


    # Đếm quốc tịch
    nationality_counts = (
        foreign_df["Nationality_Group"]
        .value_counts()
        .reset_index()
    )

    nationality_counts.columns = [
        "Nationality_Group",
        "Count"
    ]


    # 🌍 Travel / World Map Theme
    nationality_palette = [
        "#264653",  # Deep ocean navy
        "#2A9D8F",  # Asia green
        "#E9C46A",  # Desert gold
        "#F4A261",  # Sunset orange
        "#E76F51",  # Coral red
        "#8AB17D",  # Nature green
        "#577590",  # Ocean slate
        "#43AA8B",  # Emerald
        "#F9844A",  # Travel orange
        "#90BE6D",  # Forest green
        "#277DA1",  # Deep blue
        "#F94144",  # Warm red
        "#6D597A",  # Purple
        "#4D908E",  # Teal gray
        "#B56576",  # Rose
        "#355070"   # Midnight blue
    ]


    fig = px.pie(
        nationality_counts,

        names="Nationality_Group",

        values="Count",

        hole=0.45,

        color="Nationality_Group",

        color_discrete_sequence=nationality_palette
    )


    # Style pie
    fig.update_traces(

        textposition="inside",

        textinfo="label+percent",

        textfont=dict(
            size=11,
            color="white",
            family="Arial"
        ),

        marker=dict(
            line=dict(
                color="white",
                width=2
            )
        ),

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Reviews: %{value}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        )
    )


    fig.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        legend_title="Nationality",

        legend=dict(
            font=dict(
                size=11,
                color="#222222"
            )
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",


        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial"
        ),


        margin=dict(
            t=40,
            l=20,
            r=20,
            b=20
        ),

        height=400
    )


    return fig


def plot_nationality_pie_vn(df, hotel_name):
    """
    Vẽ pie chart phân bố Nationality_Group của một khách sạn cụ thể
    (bao gồm cả Việt Nam).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn được chọn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Đếm quốc tịch
    nationality_counts = (
        hotel_df["Nationality_Group"]
        .value_counts()
        .reset_index()
    )

    nationality_counts.columns = [
        "Nationality_Group",
        "Count"
    ]


    # Sắp xếp giảm dần
    nationality_counts = nationality_counts.sort_values(
        "Count",
        ascending=False
    )


    # 🌍 Travel / World Map Theme
    nationality_palette = [
        "#264653",  # Deep ocean navy
        "#2A9D8F",  # Asia green
        "#E9C46A",  # Desert gold
        "#F4A261",  # Sunset orange
        "#E76F51",  # Coral red
        "#8AB17D",  # Nature green
        "#577590",  # Ocean slate
        "#43AA8B",  # Emerald
        "#F9844A",  # Travel orange
        "#90BE6D",  # Forest green
        "#277DA1",  # Deep blue
        "#F94144",  # Warm red
        "#6D597A",  # Purple
        "#4D908E",  # Teal gray
        "#B56576",  # Rose
        "#355070"   # Midnight blue
    ]


    fig = px.pie(
        nationality_counts,

        names="Nationality_Group",

        values="Count",

        hole=0.45,

        color="Nationality_Group",

        color_discrete_sequence=nationality_palette
    )


    # Style pie
    fig.update_traces(

        # Highlight nhóm khách lớn nhất
        pull=[
            0.05 if i == 0 else 0
            for i in range(len(nationality_counts))
        ],

        textposition="inside",

        textinfo="label+percent",

        textfont=dict(
            size=11,
            color="white",
            family="Arial"
        ),

        marker=dict(
            line=dict(
                color="white",
                width=2
            )
        ),

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Guests: %{value}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        )
    )


    fig.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        legend_title="Nationality",

        legend=dict(
            font=dict(
                size=11,
                color="#222222"
            )
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",


        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial"
        ),


        margin=dict(
            t=40,
            l=20,
            r=20,
            b=20
        ),

        height=400
    )


    return fig


def plot_group_name_pie(df, hotel_name):
    """
    Vẽ pie chart phân bố Group Name của một khách sạn cụ thể.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Đếm nhóm khách
    group_counts = (
        hotel_df["Group Name"]
        .value_counts()
        .reset_index()
    )

    group_counts.columns = [
        "Group Name",
        "Count"
    ]


    # Sắp xếp giảm dần
    group_counts = group_counts.sort_values(
        "Count",
        ascending=False
    )


    # Palette đồng bộ với các biểu đồ khác
    group_colors = {
        "Cặp đôi": "#9B59B6",
        "Gia đình có em bé": "#27AE60",
        "Nhóm": "#E67E22",
        "Du lịch một mình": "#3498DB",
        "Gia đình có trẻ em": "#F1C40F",
        "Đi công tác": "#E74C3C"
    }


    fig = px.pie(
        group_counts,

        names="Group Name",

        values="Count",

        hole=0.45,

        color="Group Name",

        color_discrete_map=group_colors
    )


    # Style pie
    fig.update_traces(

        pull=[
            0.05 if i == 0 else 0
            for i in range(len(group_counts))
        ],

        textposition="inside",

        textinfo="label+percent",

        textfont=dict(
            size=11,
            color="white",
            family="Arial"
        ),

        marker=dict(
            line=dict(
                color="white",
                width=2
            )
        ),

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Guests: %{value}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        )
    )


    fig.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        legend_title="Guest Group",

        legend=dict(
            font=dict(
                size=11,
                color="#222222"
            )
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",


        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial"
        ),


        margin=dict(
            t=40,
            l=20,
            r=20,
            b=20
        ),

        height=500
    )


    return fig


def plot_group_analysis(df, hotel_name):
    """
    Vẽ:
    1. Average Stay Length theo Group Name
    2. Average Score theo Group Name

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    fig_stay : plotly.graph_objects.Figure
    fig_score : plotly.graph_objects.Figure
    """

    # Lọc khách sạn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Palette đồng bộ với Bubble Chart
    group_colors = {
        "Cặp đôi": "#9B59B6",
        "Gia đình có em bé": "#27AE60",
        "Nhóm": "#E67E22",
        "Du lịch một mình": "#3498DB",
        "Gia đình có trẻ em": "#F1C40F",
        "Đi công tác": "#E74C3C"
    }


    # ==========================
    # 1. Stay Length Analysis
    # ==========================

    stay_group = (
        hotel_df
        .groupby("Group Name")["Stay_Length"]
        .mean()
        .reset_index()
        .dropna()
    )

    stay_group.columns = [
        "Group Name",
        "Average Stay Length"
    ]

    stay_group = stay_group.sort_values(
        "Average Stay Length",
        ascending=True
    )


    fig_stay = px.bar(
        stay_group,

        x="Average Stay Length",
        y="Group Name",

        orientation="h",

        color="Group Name",

        color_discrete_map=group_colors,

        text="Average Stay Length"
    )


    fig_stay.update_traces(

        texttemplate="%{text:.2f} nights",

        textposition="outside",

        textfont=dict(
            size=11,
            color="#222222"
        ),

        marker=dict(
            opacity=0.9,
            line=dict(
                color="white",
                width=1.5
            )
        ),

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Average Stay: %{x:.2f} nights"
            "<extra></extra>"
        )
    )


    max_stay = stay_group["Average Stay Length"].max()


    fig_stay.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        xaxis_title="Average Stay Length (nights)",
        yaxis_title="",

        legend_title="Guest Group",

        plot_bgcolor="white",
        paper_bgcolor="white",


        xaxis=dict(
            range=[
                0,
                max_stay + 0.6
            ],

            title_font=dict(
                size=13,
                color="#111111"
            ),

            tickfont=dict(
                size=11,
                color="#333333"
            ),

            showgrid=True,
            gridcolor="#DDDDDD",
            gridwidth=1,
            zeroline=False
        ),


        yaxis=dict(
            tickfont=dict(
                size=11,
                color="#333333"
            )
        ),


        legend=dict(
            font=dict(
                size=11,
                color="#222222"
            )
        ),


        height=450,


        margin=dict(
            t=60,
            l=20,
            r=100,
            b=50
        )
    )


    fig_stay.update_xaxes(
        layer="below traces"
    )


    # ==========================
    # 2. Score Analysis
    # ==========================

    score_group = (
        hotel_df
        .groupby("Group Name")["Score"]
        .mean()
        .reset_index()
        .dropna()
    )

    score_group.columns = [
        "Group Name",
        "Average Score"
    ]

    score_group = score_group.sort_values(
        "Average Score",
        ascending=True
    )


    fig_score = px.bar(
        score_group,

        x="Average Score",
        y="Group Name",

        orientation="h",

        color="Group Name",

        color_discrete_map=group_colors,

        text="Average Score"
    )


    fig_score.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside",

        textfont=dict(
            size=11,
            color="#222222"
        ),

        marker=dict(
            opacity=0.9,
            line=dict(
                color="white",
                width=1.5
            )
        ),

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Average Score: %{x:.2f}"
            "<extra></extra>"
        )
    )


    min_score = score_group["Average Score"].min()
    max_score = score_group["Average Score"].max()


    fig_score.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        xaxis_title="Average Score",
        yaxis_title="",


        legend_title="Guest Group",

        plot_bgcolor="white",
        paper_bgcolor="white",


        xaxis=dict(

            range=[
                min_score - 2,
                max_score + 0.5
            ],

            title_font=dict(
                size=13,
                color="#111111"
            ),

            tickfont=dict(
                size=11,
                color="#333333"
            ),

            showgrid=True,
            gridcolor="#DDDDDD",
            gridwidth=1,
            zeroline=False
        ),


        yaxis=dict(
            tickfont=dict(
                size=11,
                color="#333333"
            )
        ),


        legend=dict(
            font=dict(
                size=11,
                color="#222222"
            )
        ),


        height=450,


        margin=dict(
            t=60,
            l=20,
            r=80,
            b=50
        )
    )


    fig_score.update_xaxes(
        layer="below traces"
    )


    return fig_stay, fig_score


def plot_group_bubble_chart(df, hotel_name):
    """
    Vẽ Bubble Chart thể hiện mối quan hệ giữa:
    - Group Name
    - Average Stay Length
    - Average Score
    - Number of Reviews

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")

    # Tổng hợp dữ liệu theo nhóm khách
    group_analysis = (
        hotel_df
        .groupby("Group Name")
        .agg(
            Average_Stay_Length=("Stay_Length", "mean"),
            Average_Score=("Score", "mean"),
            Count=("Group Name", "count")
        )
        .reset_index()
        .dropna()
    )

    if group_analysis.empty:
        raise ValueError("Không có dữ liệu Group Name.")

    # Màu sắc nổi bật hơn
    group_colors = {
        "Cặp đôi": "#9B59B6",
        "Gia đình có em bé": "#27AE60",
        "Nhóm": "#E67E22",
        "Du lịch một mình": "#3498DB",
        "Gia đình có trẻ em": "#F1C40F",
        "Đi công tác": "#E74C3C"
    }

    fig = px.scatter(
        group_analysis,

        x="Average_Stay_Length",
        y="Average_Score",

        size="Count",
        color="Group Name",

        text="Group Name",

        color_discrete_map=group_colors,

        size_max=85,

        hover_data={
            "Average_Stay_Length": ":.2f",
            "Average_Score": ":.2f",
            "Count": True
        }
    )


    # Style bubble + text
    fig.update_traces(

        textposition="top center",

        textfont=dict(
            size=11,
            color="#222222",
            family="Arial"
        ),

        marker=dict(
            opacity=0.9,
            line=dict(
                color="white",
                width=2
            )
        ),

        hovertemplate=(
            "<b>%{text}</b><br>"
            "Average Stay: %{x:.2f} nights<br>"
            "Average Score: %{y:.2f}<br>"
            "Reviews: %{marker.size}"
            "<extra></extra>"
        )
    )


    min_score = group_analysis["Average_Score"].min()


    fig.update_layout(

        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),

        xaxis_title="Average Stay Length (nights)",
        yaxis_title="Average Score",

        legend_title="Guest Group",

        plot_bgcolor="white",
        paper_bgcolor="white",


        xaxis=dict(
            title_font=dict(
                size=13,
                color="#111111"
            ),

            tickfont=dict(
                size=11,
                color="#333333"
            ),

            showgrid=True,
            gridcolor="#DDDDDD",
            gridwidth=1,

            zeroline=False
        ),


        yaxis=dict(
            title_font=dict(
                size=13,
                color="#111111"
            ),

            tickfont=dict(
                size=11,
                color="#333333"
            ),

            range=[
                min_score - 0.2,
                10.0
            ],

            showgrid=True,
            gridcolor="#DDDDDD",
            gridwidth=1,

            zeroline=False
        ),


        legend=dict(

            title_font=dict(
                size=12
            ),

            font=dict(
                size=11,
                color="#222222"
            )
        ),


        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial"
        ),


        height=600,


        margin=dict(
            t=60,
            l=50,
            r=40,
            b=50
        )
    )


    # Grid nằm sau bubble
    fig.update_xaxes(
        layer="below traces"
    )

    fig.update_yaxes(
        layer="below traces"
    )


    return fig

def plot_review_trend_by_month(df, hotel_name):
    """
    Vẽ line chart số lượng review theo tháng
    (Stay_Year + Stay_Month), bao gồm tháng không có review.
    Chỉ lấy dữ liệu từ năm 2015 trở đi.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Chuẩn bị dữ liệu thời gian
    time_df = (
        hotel_df
        .dropna(subset=["Stay_Year", "Stay_Month"])
        .copy()
    )

    time_df["Stay_Year"] = time_df["Stay_Year"].astype(int)
    time_df["Stay_Month"] = time_df["Stay_Month"].astype(int)


    # Chỉ lấy từ 2015 trở đi
    time_df = time_df[
        time_df["Stay_Year"] >= 2015
    ]


    if time_df.empty:
        raise ValueError(
            "Không có dữ liệu từ năm 2015 trở đi."
        )


    # Tạo datetime
    time_df["Stay_Date"] = pd.to_datetime(
        dict(
            year=time_df["Stay_Year"],
            month=time_df["Stay_Month"],
            day=1
        )
    )


    # Đếm review theo tháng
    review_trend = (
        time_df
        .groupby("Stay_Date")
        .size()
        .reset_index(name="Review_Count")
    )


    # Tạo đầy đủ tháng liên tục
    full_month_range = pd.date_range(
        start=review_trend["Stay_Date"].min(),
        end=review_trend["Stay_Date"].max(),
        freq="MS"
    )


    review_trend = (
        review_trend
        .set_index("Stay_Date")
        .reindex(
            full_month_range,
            fill_value=0
        )
        .rename_axis("Stay_Date")
        .reset_index()
    )


    # Line chart
    fig = px.line(
        review_trend,
        x="Stay_Date",
        y="Review_Count",
        markers=True,
        title=" "

    )


    fig.update_traces(
        line_width=3,
        marker_size=8,

        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Reviews: %{y:,}"
            "<extra></extra>"
        )
    )


    # Tạo khoảng trống dưới giá trị 0
    max_review = review_trend["Review_Count"].max()

    y_padding = max_review * 0.05


    fig.update_layout(
        title_x=0.5,

        xaxis_title="Month",
        yaxis_title="Number of Reviews",

        plot_bgcolor="white",
        paper_bgcolor="white",

        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=False
        ),

        yaxis=dict(
            range=[
                -y_padding,
                max_review * 1.1
            ],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=False
        ),

        height=450,

        margin=dict(
            t=90,
            l=50,
            r=30,
            b=50
        )
    )


    # Grid nằm phía sau line
    fig.update_xaxes(
        layer="below traces"
    )

    fig.update_yaxes(
        layer="below traces"
    )


    return fig

def plot_review_trend_by_year(df, hotel_name):
    """
    Vẽ line chart số lượng review theo năm
    (Stay_Year) của một khách sạn cụ thể.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset khách sạn
    hotel_name : str
        Tên khách sạn cần phân tích

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Lọc khách sạn
    hotel_df = df[df["Hotel_Name"] == hotel_name]

    if hotel_df.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    # Lọc dữ liệu năm
    year_df = (
        hotel_df
        .dropna(subset=["Stay_Year"])
        .copy()
    )

    year_df["Stay_Year"] = (
        year_df["Stay_Year"]
        .astype(int)
    )


    # Giới hạn từ 2015 trở đi
    year_df = year_df[
        year_df["Stay_Year"] >= 2015
    ]


    if year_df.empty:
        raise ValueError(
            "Không có dữ liệu từ năm 2015 trở đi."
        )


    # Đếm review theo năm
    review_trend = (
        year_df
        .groupby("Stay_Year")
        .size()
        .reset_index(name="Review_Count")
        .sort_values("Stay_Year")
    )


    # Vẽ line chart
    fig = px.line(
        review_trend,
        x="Stay_Year",
        y="Review_Count",
        markers=True,
        title=" "

    )


    fig.update_traces(
        line_width=3,
        marker_size=10,

        hovertemplate=(
            "<b>Year: %{x}</b><br>"
            "Reviews: %{y:,}"
            "<extra></extra>"
        )
    )


    fig.update_layout(
        title_x=0.5,

        xaxis_title="Year",
        yaxis_title="Number of Reviews",

        plot_bgcolor="white",
        paper_bgcolor="white",

        xaxis=dict(
            dtick=1,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=False
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=False
        ),

        height=450,

        margin=dict(
            t=90,
            l=50,
            r=30,
            b=50
        )
    )


    # Grid nằm sau line
    fig.update_xaxes(
        layer="below traces"
    )

    fig.update_yaxes(
        layer="below traces"
    )


    return fig



def plot_hotel_radar_new(df, hotel_name, score_cols):
    """
    Vẽ biểu đồ Radar so sánh điểm của khách sạn
    với điểm trung bình toàn hệ thống.
    """

    # Lấy dữ liệu khách sạn
    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Không tìm thấy khách sạn.")


    hotel_scores = (
        hotel.iloc[0][score_cols]
        .astype(float)
        .tolist()
    )


    system_scores = (
        df[score_cols]
        .mean()
        .tolist()
    )


    # ==========================
    # Display names
    # ==========================

    display_names = {

        "Location":
            "Location",

        "Cleanliness":
            "Cleanliness",

        "Service":
            "Service",

        "Facilities":
            "Facilities",

        "Value_for_money":
            "Value",

        "Comfort_and_room_quality":
            "Comfort"
    }


    categories = [
        display_names.get(
            col,
            col
        )
        for col in score_cols
    ]


    # Đóng polygon
    categories += [categories[0]]

    hotel_scores += [hotel_scores[0]]

    system_scores += [system_scores[0]]



    fig = go.Figure()



    # ==========================
    # Hotel
    # ==========================

    fig.add_trace(
        go.Scatterpolar(

            r=hotel_scores,

            theta=categories,

            mode="lines+markers",

            name="Khách sạn",

            line=dict(
                color="#EC4899",
                width=3
            ),

            marker=dict(
                size=12,
                color="#EC4899"
            ),

            fill="toself",

            fillcolor="rgba(236,72,153,0.20)"
        )
    )



    # ==========================
    # System Average
    # ==========================

    fig.add_trace(
        go.Scatterpolar(

            r=system_scores,

            theta=categories,

            mode="lines+markers",

            name="Trung bình Hệ thống",

            line=dict(
                color="#808080",
                width=2
            ),

            marker=dict(
                size=7,
                color="#808080"
            )
        )
    )



    fig.update_layout(

        template="plotly_white",

        height=500,


        font=dict(
            family="Arial",
            size=12,
            color="#222222"
        ),


        showlegend=True,


        margin=dict(
            l=40,
            r=40,
            t=30,
            b=30
        ),


        legend=dict(

            orientation="h",

            yanchor="bottom",
            y=1.08,

            xanchor="center",
            x=0.5,


            font=dict(
                family="Arial",
                size=12,
                color="#222222"
            )
        ),


        polar=dict(

            bgcolor="white",


            radialaxis=dict(

                visible=True,

                range=[5, 10],

                tick0=0,

                dtick=1,


                gridcolor="#D9D9D9",

                linecolor="#BFBFBF",


                tickfont=dict(
                    family="Arial",
                    size=11,
                    color="#333333"
                )
            ),


            angularaxis=dict(

                gridcolor="#D9D9D9",

                linecolor="#BFBFBF",


                tickfont=dict(
                    family="Arial",
                    size=12,
                    color="#222222"
                )
            )
        )
    )


    return fig



def plot_score_difference_new(df, hotel_name, score_cols):
    """
    Vẽ biểu đồ thanh ngang thể hiện chênh lệch điểm
    giữa khách sạn và điểm trung bình toàn hệ thống.

    Positive:
        Khách sạn cao hơn trung bình

    Negative:
        Khách sạn thấp hơn trung bình
    """

    # ==========================
    # Get hotel data
    # ==========================

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError("Không tìm thấy khách sạn.")

    hotel = hotel.iloc[0]


    # ==========================
    # Calculate difference
    # ==========================

    hotel_scores = (
        hotel[score_cols]
        .fillna(df[score_cols].mean())
        .astype(float)
    )


    avg_scores = (
        df[score_cols]
        .mean()
    )


    diff = hotel_scores - avg_scores



    # ==========================
    # Display names
    # ==========================

    display_names = {

        "Location":
            "Location",

        "Cleanliness":
            "Cleanliness",

        "Service":
            "Service",

        "Facilities":
            "Facilities",

        "Value_for_money":
            "Value",

        "Comfort_and_room_quality":
            "Comfort"
    }


    y_labels = [
        display_names.get(
            col,
            col
        )
        for col in score_cols
    ]



    # ==========================
    # Axis range
    # ==========================

    x_min = diff.min() - 0.4
    x_max = diff.max() + 0.4


    if np.isclose(x_min, x_max):
        x_min -= 0.2
        x_max += 0.2



    # ==========================
    # Travel / Premium colors
    # ==========================

    colors = [
        "#10B981" if value >= 0
        else "#F59E0B"
        for value in diff
    ]



    # ==========================
    # Create Figure
    # ==========================

    fig = go.Figure()


    fig.add_trace(

        go.Bar(

            x=diff.values,

            y=y_labels,

            orientation="h",

            width=0.6,

            marker=dict(

                color=colors,

                line=dict(
                    color="white",
                    width=1.5
                )
            ),


            text=[
                f"{value:+.2f}"
                for value in diff
            ],


            textposition="outside",


            textfont=dict(

                family="Arial",

                size=11,

                color="#222222"
            ),


            hovertemplate=(

                "<b>%{y}</b><br>"

                "Difference: %{x:+.2f} points"

                "<extra></extra>"
            )
        )
    )



    # ==========================
    # Zero line
    # ==========================

    fig.add_vline(

        x=0,

        line_color="#777777",

        line_width=1.5,

        line_dash="dash"
    )



    # ==========================
    # Layout
    # ==========================

    fig.update_layout(

        height=500,


        font=dict(

            family="Arial",

            size=12,

            color="#222222"
        ),


        plot_bgcolor="white",

        paper_bgcolor="white",


        margin=dict(

            l=40,

            r=50,

            t=20,

            b=40
        ),


        showlegend=False,


        xaxis=dict(

            title="Difference from System Average",


            title_font=dict(

                size=13,

                color="#111111"
            ),


            tickfont=dict(

                size=11,

                color="#333333"
            ),


            range=[

                x_min,

                x_max

            ],


            showgrid=True,


            gridcolor="#D9D9D9",


            zeroline=False
        ),



        yaxis=dict(

            autorange="reversed",


            tickfont=dict(

                size=12,

                color="#222222"
            ),


            showgrid=False
        )

    )


    return fig

def generate_score_difference_insight(
    df,
    hotel_name,
    score_cols,
    top_n=5
):
    """
    Tạo nhận xét tự động về điểm mạnh,
    điểm yếu và đánh giá chung của khách sạn
    dựa trên sự khác biệt với điểm trung bình hệ thống.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset hotel information

    hotel_name : str
        Tên khách sạn

    score_cols : list
        Các cột điểm đánh giá

    top_n : int
        Số lượng tiêu chí nổi bật muốn lấy

    Returns
    -------
    dict
        {
            "strength": str,
            "weakness": str,
            "summary": str
        }
    """


    # ==========================
    # Get hotel
    # ==========================

    hotel = df[df["Hotel_Name"] == hotel_name]

    if hotel.empty:
        raise ValueError(
            "Không tìm thấy khách sạn."
        )

    hotel = hotel.iloc[0]


    # ==========================
    # Calculate difference
    # ==========================

    hotel_scores = (
        hotel[score_cols]
        .fillna(df[score_cols].mean())
        .astype(float)
    )


    avg_scores = (
        df[score_cols]
        .mean()
    )


    diff = (
        hotel_scores - avg_scores
    )


    # Rename
    display_names = {

        "Location":
            "Location",

        "Cleanliness":
            "Cleanliness",

        "Service":
            "Service",

        "Facilities":
            "Facilities",

        "Value_for_money":
            "Value",

        "Comfort_and_room_quality":
            "Comfort"

    }


    diff.index = [
        display_names.get(
            x,
            x
        )
        for x in diff.index
    ]



    # ==========================
    # Strength / Weakness
    # ==========================

    strengths = (
        diff
        .sort_values(
            ascending=False
        )
        .head(top_n)
    )


    weaknesses = (
        diff
        .sort_values(
            ascending=True
        )
        .head(top_n)
    )


    # ==========================
    # Generate text
    # ==========================

    strength_text = ", ".join(
        [
            f"{name} (+{value:.2f})"
            for name, value in strengths.items()
            if value > 0
        ]
    )


    weakness_text = ", ".join(
        [
            f"{name} ({value:.2f})"
            for name, value in weaknesses.items()
            if value < 0
        ]
    )


    # ==========================
    # Overall judgement
    # ==========================

    positive_count = (
        (diff > 0)
        .sum()
    )

    negative_count = (
        (diff < 0)
        .sum()
    )


    avg_difference = diff.mean()



    if avg_difference >= 0.3:

        summary = (
            "Khách sạn đang có hiệu suất "
            "vượt trội so với mặt bằng chung. "
            "Các tiêu chí đánh giá nhìn chung "
            "đang tạo lợi thế cạnh tranh."
        )


    elif avg_difference >= 0:

        summary = (
            "Khách sạn có chất lượng tương đương "
            "hoặc hơi cao hơn mức trung bình. "
            "Một số điểm mạnh có thể tiếp tục "
            "được phát huy để tạo khác biệt."
        )


    elif avg_difference > -0.3:

        summary = (
            "Chất lượng tổng thể đang gần với "
            "mức trung bình của hệ thống. "
            "Khách sạn nên tập trung cải thiện "
            "các tiêu chí có điểm thấp hơn."
        )


    else:

        summary = (
            "Khách sạn đang thấp hơn đáng kể "
            "so với mặt bằng chung. "
            "Cần ưu tiên cải thiện các yếu tố "
            "ảnh hưởng trực tiếp đến trải nghiệm khách."
        )



    # ==========================
    # Empty handling
    # ==========================

    if strength_text == "":
        strength_text = (
            "Chưa có tiêu chí nào vượt "
            "mức trung bình rõ rệt."
        )


    if weakness_text == "":
        weakness_text = (
            "Không phát hiện tiêu chí "
            "thấp hơn mức trung bình."
        )



    return {

        "strength":
            f"🌸 Điểm mạnh: {strength_text}",

        "weakness":
            f"🌿 Điểm cần cải thiện: {weakness_text}",

        "summary":
            f"📊 Đánh giá chung: {summary}"

    }