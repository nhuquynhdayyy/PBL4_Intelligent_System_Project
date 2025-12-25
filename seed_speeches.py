import random
from app import app, db
from database import Student, Subject, Session, SpeechLog
from datetime import datetime, timedelta

def seed_data():
    with app.app_context():
        print("--- ĐANG TẠO DỮ LIỆU PHÁT BIỂU MẪU ---")
        
        # 1. Lấy danh sách học sinh và môn học
        students = Student.query.all()
        subjects = Subject.query.all()
        
        if not students or not subjects:
            print("Lỗi: Bạn cần có ít nhất 1 học sinh và 1 môn học trong DB trước.")
            return

        # 2. Tạo các buổi học (Sessions) cho mỗi môn
        # Mỗi môn tạo khoảng 5 buổi học để có dữ liệu dày
        all_sessions = []
        for sub in subjects:
            for i in range(2):
                sess = Session(
                    subject_id=sub.id,
                    class_id=sub.class_id,
                    status='ended',
                    end_time=datetime.utcnow() - timedelta(days=random.randint(1, 30))
                )
                db.session.add(sess)
                all_sessions.append(sess)
        
        db.session.commit() # Lưu session để lấy ID
        print(f"Đã tạo {len(all_sessions)} buổi học mẫu.")

        # 3. Tạo lượt phát biểu (SpeechLogs) theo kịch bản để test K-Means
        # Kịch bản:
        # - 1/3 số học sinh đầu tiên: Phát biểu nhiều các môn Tự nhiên
        # - 1/3 số học sinh tiếp theo: Phát biểu nhiều các môn Xã hội
        # - 1/3 còn lại: Rất ít phát biểu (thụ động)

        count = 0
        for i, stu in enumerate(students):
            # Xác định "gu" của học sinh dựa trên chỉ số i
            if i % 3 == 0:
                pattern = "KHTN"
            elif i % 3 == 1:
                pattern = "KHXH"
            else:
                pattern = "PASSIVE"

            for sess in all_sessions:
                sub = Subject.query.get(sess.subject_id)
                
                # Quyết định số lần phát biểu trong buổi này
                num_speeches = 0
                if pattern == "KHTN":
                    if sub.category == "Khoa học Tự nhiên":
                        num_speeches = random.randint(2, 5) # Phát biểu nhiều
                    else:
                        num_speeches = random.randint(0, 1)
                
                elif pattern == "KHXH":
                    if sub.category == "Khoa học Xã hội" or sub.category == "Ngoại ngữ":
                        num_speeches = random.randint(2, 5)
                    else:
                        num_speeches = random.randint(0, 1)
                
                else: # PASSIVE
                    num_speeches = random.randint(0, 1) if random.random() > 0.7 else 0

                # Thêm vào DB
                for _ in range(num_speeches):
                    log = SpeechLog(student_id=stu.id, session_id=sess.id)
                    db.session.add(log)
                    count += 1

        db.session.commit()
        print(f"Thành công! Đã thêm {count} lượt phát biểu mẫu cho {len(students)} học sinh.")
        print("Bây giờ bạn có thể chạy 'python analyze_trends.py' để train.")

if __name__ == "__main__":
    seed_data()