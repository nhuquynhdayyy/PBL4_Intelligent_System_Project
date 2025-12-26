from app import app, db
from database import Session, SpeechLog

with app.app_context():
    # Xóa toàn bộ lượt phát biểu
    db.session.query(SpeechLog).delete()
    # Xóa toàn bộ buổi học mẫu đã tạo
    db.session.query(Session).delete()
    db.session.commit()
    print("Đã xóa sạch dữ liệu phát biểu và buổi học cũ!")