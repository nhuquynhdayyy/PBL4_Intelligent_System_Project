import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json, os
from dotenv import load_dotenv

# --- KẾT NỐI DATABASE ---
load_dotenv()
db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', '')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '3307') 
db_name = os.getenv('DB_NAME', 'pbl4_db')

engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

def train_behavioral_ai():
    print("=== ĐANG PHÂN TÍCH PHONG CÁCH HỌC TẬP (AI K-MEANS) ===")
    
    # Trích xuất dữ liệu hành vi tổng quát
    query = """
    SELECT 
        s.id as student_id,
        COUNT(DISTINCT sl.id) as total_speech,
        AVG(g.score) as avg_grade
    FROM students s
    LEFT JOIN speech_logs sl ON s.id = sl.student_id
    LEFT JOIN grades g ON s.id = g.student_id
    GROUP BY s.id
    """
    df = pd.read_sql(query, engine).fillna(0)
    
    if df.empty:
        print("[LỖI] Không có dữ liệu để huấn luyện.")
        return

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features = df[['total_speech', 'avg_grade']]
    scaled_features = scaler.fit_transform(features)

    # Chạy K-Means tìm 4 nhóm phong cách
    kmeans = KMeans(n_clusters=min(4, len(df)), random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # Phân tích đặc điểm trung bình của từng cụm để dán nhãn sư phạm
    cluster_centers = df.groupby('cluster')[['total_speech', 'avg_grade']].mean()
    style_map = {}

    for cluster_id, row in cluster_centers.iterrows():
        speech = row['total_speech']
        grade = row['avg_grade']
        
        # Logic dán nhãn chuẩn môi trường học đường
        if grade >= 8.0 and speech >= 15:
            style = "Học sinh Năng động & Xuất sắc"
            advice = "Em là tấm gương về sự kết hợp giữa kiến thức và sự chủ động. Hãy tiếp tục phát huy vai trò dẫn dắt trong các hoạt động nhóm."
        elif grade >= 8.0:
            style = "Học sinh Chuyên sâu & ít thể hiện"
            advice = "Em có năng lực tiếp thu rất tốt. Nếu em chủ động chia sẻ ý kiến nhiều hơn, em sẽ tạo được sức ảnh hưởng lớn hơn tới tập thể."
        elif grade >= 5.0 and speech >= 15:
            style = "Học sinh Năng nổ nhưng chưa bứt phá"
            advice = "Thái độ học tập của em rất đáng khen ngợi. Hãy tập trung rà soát lại phương pháp tự học để tối ưu hóa kết quả điểm số cao hơn."
        else:
            style = "Học sinh Thụ động & Cần hỗ trợ"
            advice = "Mỗi cá nhân đều có thế mạnh riêng. Em cần tự tin hơn và chủ động tương tác với giáo viên để tháo gỡ khó khăn trong bài học."
        
        style_map[int(cluster_id)] = {"style": style, "advice": advice}

    # Lưu kết quả vào file JSON để Backend app.py đọc
    final_analysis = {}
    for _, row in df.iterrows():
        final_analysis[str(int(row['student_id']))] = style_map[int(row['cluster'])]

    with open('behavioral_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(final_analysis, f, ensure_ascii=False, indent=4)
    
    print(f"--- THÀNH CÔNG! ĐÃ CẬP NHẬT PHONG CÁCH CHO {len(df)} HỌC SINH ---")

if __name__ == '__main__':
    train_behavioral_ai()