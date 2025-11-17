import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
import json # Thêm thư viện json
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
    # ... (Hàm này giữ nguyên như cũ, không cần thay đổi)
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
    # ... (Hàm này giữ nguyên như cũ, không cần thay đổi)
    features = df.drop(columns=['student_id', 'full_name'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Tăng n_clusters lên 4 để phù hợp với 4 nhóm dữ liệu giả
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans.fit(features_scaled)
    df['cluster'] = kmeans.labels_
    joblib.dump(kmeans, 'student_cluster_model.pkl')
    joblib.dump(scaler, 'student_data_scaler.pkl')
    print("\n[INFO] Đã lưu model và scaler vào file .pkl")
    return df

# CẬP NHẬT LỚN: Hàm này giờ sẽ tạo ra bản đồ động
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