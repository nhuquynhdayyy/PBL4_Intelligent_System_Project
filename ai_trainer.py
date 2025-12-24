import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import joblib
import os
import json
from dotenv import load_dotenv

# --- PHẦN 1: KẾT NỐI DATABASE ---
load_dotenv()
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')
DATABASE_URI = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
engine = create_engine(DATABASE_URI)

def prepare_data():
    query = """
    SELECT
        s.id AS student_id, s.full_name,
        COUNT(DISTINCT CASE WHEN sub.category = 'Khoa học Tự nhiên' THEN sl.id END) AS speeches_natural_science,
        COUNT(DISTINCT CASE WHEN sub.category = 'Khoa học Xã hội' THEN sl.id END) AS speeches_social_science,
        COUNT(DISTINCT CASE WHEN sub.category = 'Ngoại ngữ' THEN sl.id END) AS speeches_language,
        COUNT(DISTINCT CASE WHEN sub.category = 'Năng khiếu' THEN sl.id END) AS speeches_aptitude,
        AVG(CASE WHEN sub.category = 'Khoa học Tự nhiên' THEN g.score END) AS avg_grade_natural_science,
        AVG(CASE WHEN sub.category = 'Khoa học Xã hội' THEN g.score END) AS avg_grade_social_science,
        AVG(CASE WHEN sub.category = 'Ngoại ngữ' THEN g.score END) AS avg_grade_language,
        AVG(CASE WHEN sub.category = 'Năng khiếu' THEN g.score END) AS avg_grade_aptitude
    FROM students s
    LEFT JOIN speech_logs sl ON s.id = sl.student_id
    LEFT JOIN sessions sess ON sl.session_id = sess.id
    LEFT JOIN grades g ON s.id = g.student_id
    LEFT JOIN subjects sub ON sub.id = sess.subject_id OR sub.id = g.subject_id
    WHERE s.class_id = 1
    GROUP BY s.id, s.full_name;
    """
    df = pd.read_sql(query, engine)
    df_cleaned = df.fillna(0)
    return df_cleaned

def train_model(df):
    features = df.drop(columns=['student_id', 'full_name'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    training_history = []
    ks = range(1, 11) # Bắt đầu từ 1 thay vì 2
    
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(features_scaled)
        
        inertia = km.inertia_
        
        # Xử lý riêng cho K=1 để tránh lỗi Silhouette
        if k > 1:
            sil = silhouette_score(features_scaled, km.labels_)
        else:
            sil = 0 # Hoặc để trống (None) cho K=1
            
        training_history.append({
            'K': k,
            'Inertia': inertia,
            'Silhouette': sil
        })

    # --- XUẤT EXCEL CÓ BIỂU ĐỒ (DÙNG SCATTER ĐỂ CÓ SỐ 0) ---
    history_df = pd.DataFrame(training_history)
    file_name = "clustering_training_report.xlsx"
    
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    history_df.to_excel(writer, sheet_name='Training_Log', index=False)
    
    workbook  = writer.book
    worksheet = writer.sheets['Training_Log']

    # 1. Vẽ biểu đồ Elbow (Sử dụng Scatter để trục X là số và bắt đầu từ 0)
    chart_elbow = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_elbow.add_series({
        'name':       'Inertia Loss (Elbow Method)',
        'categories': ['Training_Log', 1, 0, 10, 0], # Cột K (Trục X) - Index từ dòng 1 đến 10
        'values':     ['Training_Log', 1, 1, 10, 1], # Cột Inertia (Trục Y)
        'marker':     {'type': 'circle', 'size': 8, 'border': {'color': 'blue'}, 'fill': {'color': 'blue'}},
        'line':       {'color': 'blue'},
    })
    chart_elbow.set_title({'name': 'Biểu đồ Elbow tìm K tối ưu'})
    chart_elbow.set_x_axis({
        'name': 'Số lượng cụm (K)',
        'min': 0,           # BẮT ĐẦU TỪ 0
        'max': 10,          # KẾT THÚC Ở 10
        'major_unit': 1,    # Chia vạch mỗi 1 đơn vị
    })
    chart_elbow.set_y_axis({'name': 'Inertia (Loss)'})
    
    # 2. Vẽ biểu đồ Silhouette
    chart_sil = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_sil.add_series({
        'name':       'Silhouette Score',
        'categories': ['Training_Log', 1, 0, 10, 0], # Cột K (Trục X)
        'values':     ['Training_Log', 1, 2, 10, 2], # Cột Silhouette (Trục Y)
        'line':       {'color': 'red'},
        'marker':     {'type': 'square', 'size': 8, 'border': {'color': 'red'}, 'fill': {'color': 'red'}},
    })
    chart_sil.set_title({'name': 'Biểu đồ Silhouette Score (Càng cao càng tốt)'})
    chart_sil.set_x_axis({
        'name': 'Số lượng cụm (K)',
        'min': 0, 
        'max': 10,
        'major_unit': 1,
    })
    chart_sil.set_y_axis({'name': 'Silhouette Score'})

    # Chèn biểu đồ vào sheet
    worksheet.insert_chart('E2', chart_elbow)
    worksheet.insert_chart('E18', chart_sil)

    writer.close()
    print(f"\n[SUCCESS] Đã xuất file {file_name} kèm biểu đồ tự động.")

    # Huấn luyện model cuối cùng
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans.fit(features_scaled)
    df['cluster'] = kmeans.labels_
    joblib.dump(kmeans, 'student_cluster_model.pkl')
    joblib.dump(scaler, 'student_data_scaler.pkl')
    return df

# Hàm tạo ra bản đồ động
def analyze_and_create_map(df_clustered):
    """
    Phân tích đặc điểm của các cụm và tạo ra một file map (JSON)
    để ánh xạ ID cụm sang nhãn có ý nghĩa.
    """
    if 'cluster' not in df_clustered.columns:
        print("Lỗi: Dữ liệu chưa được phân cụm.")
        return

    # Tính các giá trị trung bình của các đặc trưng chính cho mỗi cụm
    cluster_analysis = df_clustered.groupby('cluster')[['avg_grade_natural_science', 'avg_grade_social_science', 'avg_grade_language', 'avg_grade_aptitude']].mean()
    
    print("\n--- BẢNG PHÂN TÍCH TRUNG BÌNH CÁC CỤM ---")
    print(cluster_analysis)

    # Tự động xác định ý nghĩa của mỗi cụm
    cluster_map = {}
    # Tìm ra cột có giá trị cao nhất cho mỗi cụm (mỗi hàng)
    dominant_feature = cluster_analysis.idxmax(axis=1)

    # Tạo nhãn dựa trên đặc trưng nổi trội
    for cluster_id, feature_name in dominant_feature.items():
        if 'natural_science' in feature_name:
            label = "Thiên về Khoa học Tự nhiên"
        elif 'social_science' in feature_name:
            label = "Thiên về Khoa học Xã hội"
        elif 'language' in feature_name:
            label = "Thiên về Ngoại ngữ"
        elif 'aptitude' in feature_name:
            label = "Thiên về Năng khiếu"
        else:
            label = "Phát triển cân bằng"
        cluster_map[cluster_id] = label
    
    # Xử lý trường hợp có các cụm có điểm thấp đều (học lực trung bình)
    # Ví dụ: nếu tổng điểm trung bình của một cụm là thấp nhất, gán là "Cần cố gắng"
    cluster_sums = cluster_analysis.sum(axis=1)
    if len(cluster_sums) > 2: # Chỉ áp dụng nếu có nhiều hơn 2 cụm
        weakest_cluster = cluster_sums.idxmin()
        cluster_map[weakest_cluster] = "Cần cố gắng thêm"

    print("\n--- BẢN ĐỒ CỤM ĐƯỢC TẠO TỰ ĐỘNG ---")
    print(cluster_map)

    # Lưu bản đồ này vào file JSON
    with open('cluster_map.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_map, f, ensure_ascii=False, indent=4)
    
    print("\n[INFO] Đã lưu bản đồ cụm vào file cluster_map.json")


# --- Chạy hàm chính ---
if __name__ == '__main__':
    print("Bắt đầu Giai đoạn 1: Trích xuất và chuẩn bị dữ liệu...")
    student_data = prepare_data()
    
    if not student_data.empty:
        print("\nBắt đầu Giai đoạn 2: Huấn luyện mô hình...")
        clustered_data = train_model(student_data)
        
        print("\nBắt đầu Giai đoạn 3: Phân tích và tạo bản đồ cụm...")
        analyze_and_create_map(clustered_data)
        
        print("\n[THÀNH CÔNG] Toàn bộ quá trình huấn luyện và tạo bản đồ đã hoàn tất.")
    else:
        print("\n[CẢNH BÁO] Không có dữ liệu để huấn luyện.")