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
    # Tăng n_clusters lên 4 để phù hợp với 4 nhóm dữ liệu
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans.fit(features_scaled)
    df['cluster'] = kmeans.labels_
    # Ghi thẳng kết quả phân cụm vào Database để Web đọc
    try:
        df.to_sql('student_clusters', engine, if_exists='replace', index=False)
        print("[INFO] Đã cập nhật kết quả phân cụm vào bảng 'student_clusters' trong DB")
    except Exception as e:
        print(f"[LỖI] Không thể ghi vào DB: {e}")
    # ----------------------------------------
    joblib.dump(kmeans, 'student_cluster_model.pkl')
    joblib.dump(scaler, 'student_data_scaler.pkl')
    print("\n[INFO] Đã lưu model và scaler vào file .pkl")
    return df

def analyze_and_create_map(df_clustered):
    # 1. Tính trung bình các môn
    cluster_analysis = df_clustered.groupby('cluster')[['avg_grade_natural_science', 'avg_grade_social_science', 'avg_grade_language', 'avg_grade_aptitude']].mean()
    
    cluster_map = {}
    
    for cluster_id, row in cluster_analysis.iterrows():
        # Lấy các giá trị điểm
        scores = {
            "Khoa học Tự nhiên": row['avg_grade_natural_science'],
            "Khoa học Xã hội": row['avg_grade_social_science'],
            "Ngoại ngữ": row['avg_grade_language'],
            "Năng khiếu": row['avg_grade_aptitude']
        }
        
        # Tìm môn cao nhất
        best_subject = max(scores, key=scores.get)
        max_score = scores[best_subject]
        
        # --- LOGIC MỚI ---
        # 1. Nếu điểm cao nhất mà vẫn dưới 5.0 -> Cần cố gắng
        if max_score < 5.0:
            label = "Cần cố gắng thêm"
        # 2. Nếu các môn lệch nhau ít (dưới 0.7 điểm) -> Phát triển cân bằng
        elif max(scores.values()) - min([v for v in scores.values() if v > 0]) < 0.7:
            label = "Phát triển cân bằng"
        # 3. Còn lại dán nhãn theo môn cao nhất
        else:
            label = f"Thiên về {best_subject}"
            
        cluster_map[cluster_id] = label

    print("\n--- BẢN ĐỒ CỤM ĐÃ ĐƯỢC SỬA LỖI ---")
    print(cluster_map)

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