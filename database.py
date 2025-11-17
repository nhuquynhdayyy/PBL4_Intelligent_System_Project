# database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Class(db.Model):
    """Bảng lưu thông tin các lớp học."""
    __tablename__ = 'classes'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False, comment="Tên lớp, ví dụ: 9/1")
    academic_year = db.Column(db.String(50), nullable=True, comment="Năm học, ví dụ: 2025-2026")

    # Mối quan hệ
    students = db.relationship('Student', backref='class_info', lazy=True)
    subjects = db.relationship('Subject', backref='class_info', lazy=True)

class Student(db.Model):
    """Bảng lưu thông tin học sinh."""
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    student_code = db.Column(db.String(50), unique=True, nullable=False, comment="Mã định danh, trùng với tên thư mục ảnh")
    full_name = db.Column(db.String(100), nullable=False)
    # --- CẬP NHẬT: Thêm thông tin chi tiết cho học sinh ---
    date_of_birth = db.Column(db.Date, nullable=True, comment="Ngày sinh của học sinh")
    gender = db.Column(db.String(10), nullable=True, comment="Giới tính: Nam, Nữ, Khác")
    
    # Khóa ngoại
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)

    # Mối quan hệ
    speech_logs = db.relationship('SpeechLog', backref='student', lazy='dynamic')
    # --- CẬP NHẬT: Thêm mối quan hệ với bảng điểm ---
    grades = db.relationship('Grade', backref='student', lazy=True)

class Subject(db.Model):
    """Bảng lưu thông tin các môn học."""
    __tablename__ = 'subjects'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    icon = db.Column(db.String(10), nullable=True, comment="Biểu tượng emoji")
    # --- CẬP NHẬT: Thêm phân loại môn học ---
    category = db.Column(db.String(100), nullable=True, comment="Phân loại môn học: Tự nhiên, Xã hội,...")
    
    # Khóa ngoại
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    
    # Mối quan hệ
    sessions = db.relationship('Session', backref='subject', lazy=True)
    # --- CẬP NHẬT: Thêm mối quan hệ với bảng điểm ---
    grades = db.relationship('Grade', backref='subject', lazy=True)

class Session(db.Model):
    """Bảng lưu thông tin mỗi buổi học."""
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='ongoing', comment="Trạng thái: ongoing, ended")
    
    # Khóa ngoại
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    
    # Mối quan hệ
    speech_logs = db.relationship('SpeechLog', backref='session', lazy='dynamic')

class SpeechLog(db.Model):
    """Bảng ghi lại mỗi lượt phát biểu."""
    __tablename__ = 'speech_logs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    confidence = db.Column(db.Float, nullable=True)
    
    # Khóa ngoại
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)

# --- MỚI: Bảng lưu điểm số của học sinh ---
class Grade(db.Model):
    """Bảng lưu điểm số của học sinh."""
    __tablename__ = 'grades'
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Float, nullable=False, comment="Điểm số")
    grade_type = db.Column(db.String(50), nullable=True, comment="Loại điểm: 15 phút, 1 tiết, học kỳ,...")
    term = db.Column(db.String(50), nullable=True, comment="Học kỳ: I, II, Hè")
    exam_date = db.Column(db.Date, default=datetime.utcnow, comment="Ngày kiểm tra")

    # Khóa ngoại
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)