// static/js/students.js
document.addEventListener('DOMContentLoaded', function() {
    const studentList = document.getElementById('studentList');
    const studentForm = document.getElementById('studentForm');
    const addStudentBtn = document.getElementById('addStudentBtn');

    async function loadStudentsData() {
        const students = await apiCall('/api/students');
        if (!students) { studentList.innerHTML = '<p>Lỗi tải dữ liệu học sinh.</p>'; return; }
        
        if (students.length === 0) {
            studentList.innerHTML = '<p>Chưa có học sinh nào. Nhấn "Thêm học sinh mới" để bắt đầu.</p>';
            return;
        }
        
        studentList.innerHTML = students.map(stu => `
            <div class="card">
                <div style="display: flex; align-items: center; flex-grow: 1;">
                    <div class="student-avatar">${stu.full_name.charAt(0)}</div>
                    <div>
                        <strong>${stu.full_name}</strong>
                        <p style="color: #666; font-size: 14px;">Face ID: ${stu.student_code} | Tổng: ${stu.total_speeches} phát biểu</p>
                    </div>
                </div>
                <div>
                    <button class="btn btn-action" data-action="edit-student" data-id="${stu.id}">Sửa</button>
                    <button class="btn btn-danger btn-action" data-action="delete-student" data-id="${stu.id}" data-name="${stu.full_name}">Xóa</button>
                </div>
            </div>
        `).join('');
    }
    
    if (studentForm) {
        studentForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const id = document.getElementById('studentId').value;
            const data = { name: document.getElementById('studentName').value };
            if (!id) data.code = document.getElementById('studentCode').value;

            const endpoint = id ? `/api/students/${id}` : '/api/students';
            const method = id ? 'PUT' : 'POST';
            
            if (!id && !data.code) { alert("Vui lòng chọn một Face ID!"); return; }

            const result = await apiCall(endpoint, method, data);
            if (result) { alert(result.message); closeModal('studentModal'); loadStudentsData(); }
        });
    }

    if (addStudentBtn) {
        addStudentBtn.addEventListener('click', async () => {
            studentForm.reset();
            document.getElementById('studentModalTitle').textContent = '➕ Thêm học sinh mới';
            document.getElementById('studentId').value = '';
            document.querySelector('#studentCode').parentElement.style.display = 'block';
            
            const studentCodeSelect = document.getElementById('studentCode');
            studentCodeSelect.innerHTML = '<option value="">-- Đang tải Face ID... --</option>';
            const availableCodes = await apiCall('/api/untrained_faces');
            if (availableCodes) {
                studentCodeSelect.innerHTML = availableCodes.length > 0
                    ? '<option value="">-- Chọn Face ID --</option>' + availableCodes.map(c => `<option value="${c}">${c}</option>`).join('')
                    : '<option value="" disabled>-- Không có Face ID mới --</option>';
            }
            openModal('studentModal');
        });
    }

    document.body.addEventListener('click', async (e) => {
        const target = e.target.closest('.btn-action[data-action]');
        if (!target) return;
        const action = target.dataset.action;
        const id = target.dataset.id;
        const name = target.dataset.name;

        if (action === 'edit-student') {
            const student = await apiCall(`/api/students/${id}`);
            if (!student) return;
            document.getElementById('studentModalTitle').textContent = `✏️ Chỉnh sửa học sinh`;
            document.getElementById('studentId').value = student.id;
            document.getElementById('studentName').value = student.full_name;
            document.querySelector('#studentCode').parentElement.style.display = 'none'; // Ẩn chọn Face ID khi sửa
            openModal('studentModal');
        } else if (action === 'delete-student') {
            if (confirm(`Bạn có chắc muốn xóa học sinh "${name}"?`)) {
                const result = await apiCall(`/api/students/${id}`, 'DELETE');
                if (result) { alert(result.message); loadStudentsData(); }
            }
        }
    });

    loadStudentsData();
});