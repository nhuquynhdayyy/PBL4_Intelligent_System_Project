# database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Class(db.Model):
    """Bảng lưu thông tin các lớp học."""
    __tablename__ = 'classes'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False, comment="Tên lớp, ví dụ: 10/1")
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
    
    # Khóa ngoại
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)

    # Mối quan hệ
    speech_logs = db.relationship('SpeechLog', backref='student', lazy='dynamic') # lazy='dynamic' cho phép query thêm

class Subject(db.Model):
    """Bảng lưu thông tin các môn học."""
    __tablename__ = 'subjects'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    icon = db.Column(db.String(10), nullable=True, comment="Biểu tượng emoji")
    
    # Khóa ngoại
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    
    # Mối quan hệ
    sessions = db.relationship('Session', backref='subject', lazy=True)

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