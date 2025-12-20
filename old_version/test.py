#Mở tệp pickle chứa cơ sở dữ liệu khuôn mặt, in ra mỗi ID sinh viên cùng với độ dài của embedding khuôn mặt tương ứng.
import pickle
f = open("face_db.pkl","rb")
db = pickle.load(f)
f.close()

for sid, emb_list in db.items():
    print(sid, len(emb_list[0]))  # xem độ dài embedding 
