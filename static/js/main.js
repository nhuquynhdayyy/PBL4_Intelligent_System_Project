// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {

    // =============================================================
    // === KHỐI HÀM TIỆN ÍCH (HELPER FUNCTIONS) ===
    // =============================================================
    async function apiCall(endpoint, method = 'GET', body = null) {
        const options = { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : null };
        try {
            const response = await fetch(endpoint, options);
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || 'Có lỗi không xác định.');
            return data;
        } catch (error) {
            console.error(`Lỗi API ${endpoint}:`, error);
            alert(`Lỗi: ${error.message}`);
            return null;
        }
    }

    function openModal(modalId) { document.getElementById(modalId).classList.add('show'); }
    function closeModal(modalId) { document.getElementById(modalId).classList.remove('show'); }

    // =============================================================
    // === CÁC HÀM TẢI VÀ RENDER DỮ LIỆU ===
    // =============================================================
    async function loadSubjectsData() {
        const subjects = await apiCall('/api/subjects');
        const subjectGrid = document.getElementById('subjectGrid');
        if (!subjects) { subjectGrid.innerHTML = '<p>Lỗi tải dữ liệu môn học.</p>'; return; }
        if (subjects.length === 0) {
            subjectGrid.innerHTML = '<p>Chưa có môn học nào. Nhấn "Thêm môn học mới" để bắt đầu.</p>';
            return;
        }
        subjectGrid.innerHTML = subjects.map(sub => `
            <div class="subject-card">
                <h3>${sub.icon || '📚'} ${sub.name}</h3>
                <p>Đã dạy: ${sub.session_count} buổi</p>
                <p>Tổng phát biểu: ${sub.total_speeches} lần</p>
                <div class="subject-actions">
                    <button class="btn btn-action" data-action="start-session" data-id="${sub.id}"><i class="fa-solid fa-play"></i> Mở lớp</button>
                    <button class="btn btn-action" data-action="history" data-id="${sub.id}" data-name="${sub.name}"><i class="fa-solid fa-clock-rotate-left"></i> Lịch sử</button>
                    <button class="btn btn-action" data-action="edit-subject" data-id="${sub.id}"><i class="fa-solid fa-pen"></i></button>
                    <button class="btn btn-danger btn-action" data-action="delete-subject" data-id="${sub.id}" data-name="${sub.name}"><i class="fa-solid fa-trash"></i></button>
                </div>
            </div>
        `).join('');
    }
    
    async function loadStudentsData() {
         const students = await apiCall('/api/students');
         const studentList = document.getElementById('studentList');
         if (!students) { studentList.innerHTML = '<p>Lỗi tải dữ liệu học sinh.</p>'; return; }
         if (students.length === 0) {
            studentList.innerHTML = '<p>Chưa có học sinh nào. Nhấn "Thêm học sinh mới" để bắt đầu.</p>';
            return;
         }
         studentList.innerHTML = students.map(stu => `
            <div class="card">
                <div style="display: flex; align-items: center;">
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
    
    // =============================================================
    // === XỬ LÝ SỰ KIỆN VÀ LOGIC CRUD ===
    // =============================================================
    
    // --- Quản lý Modal ---
    document.querySelectorAll('.modal-close, .modal button[type="button"].btn-danger').forEach(el => {
        el.addEventListener('click', () => closeModal(el.closest('.modal').id));
    });

    // --- CRUD Môn học (Form Submit) ---
    const subjectForm = document.getElementById('subjectForm');
    if (subjectForm) {
        subjectForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const id = document.getElementById('subjectId').value;
            const data = {
                name: document.getElementById('subjectName').value,
                icon: document.getElementById('subjectIcon').value
            };
            const endpoint = id ? `/api/subjects/${id}` : '/api/subjects';
            const method = id ? 'PUT' : 'POST';
            
            const result = await apiCall(endpoint, method, data);
            if (result) {
                alert(result.message);
                closeModal('subjectModal');
                loadSubjectsData();
            }
        });
    }

    // --- CRUD Học sinh (Form Submit) ---
    const studentForm = document.getElementById('studentForm');
    if (studentForm) {
        studentForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const id = document.getElementById('studentId').value;
            const data = { name: document.getElementById('studentName').value };
            if (!id) { data.code = document.getElementById('studentCode').value; }

            const endpoint = id ? `/api/students/${id}` : '/api/students';
            const method = id ? 'PUT' : 'POST';

            if (!id && !data.code) { alert("Vui lòng chọn một Face ID!"); return; }

            const result = await apiCall(endpoint, method, data);
            if (result) {
                alert(result.message);
                closeModal('studentModal');
                loadStudentsData();
            }
        });
    }

    // --- Sử dụng Event Delegation cho các nút Sửa/Xóa/Lịch sử ---
    document.body.addEventListener('click', async (e) => {
        const target = e.target.closest('.btn-action');
        if (!target) return;

        const action = target.dataset.action;
        const id = target.dataset.id;
        const name = target.dataset.name;

        if (action === 'edit-subject') {
            const subject = await apiCall(`/api/subjects/${id}`);
            if (!subject) return;
            document.getElementById('subjectModalTitle').textContent = `✏️ Chỉnh sửa môn học`;
            document.getElementById('subjectId').value = subject.id;
            document.getElementById('subjectName').value = subject.name;
            document.getElementById('subjectIcon').value = subject.icon;
            openModal('subjectModal');
        }
        if (action === 'delete-subject') {
            if (confirm(`Bạn có chắc muốn xóa môn học "${name}"?\nMọi dữ liệu liên quan sẽ bị mất!`)) {
                const result = await apiCall(`/api/subjects/${id}`, 'DELETE');
                if (result) { alert(result.message); loadSubjectsData(); }
            }
        }
        if (action === 'edit-student') {
            const student = await apiCall(`/api/students/${id}`);
            if (!student) return;
            document.getElementById('studentModalTitle').textContent = `✏️ Chỉnh sửa học sinh`;
            document.getElementById('studentId').value = student.id;
            document.getElementById('studentName').value = student.full_name;
            document.querySelector('#studentCode').parentElement.style.display = 'none';
            openModal('studentModal');
        }
        if (action === 'delete-student') {
            if (confirm(`Bạn có chắc muốn xóa học sinh "${name}"?`)) {
                const result = await apiCall(`/api/students/${id}`, 'DELETE');
                if (result) { alert(result.message); loadStudentsData(); }
            }
        }
    });
    
    // --- GÁN SỰ KIỆN CHO CÁC NÚT "THÊM MỚI" BẰNG ID ---
    const addSubjectBtn = document.getElementById('addSubjectBtn');
    if (addSubjectBtn) {
        addSubjectBtn.addEventListener('click', () => {
            subjectForm.reset();
            document.getElementById('subjectModalTitle').textContent = '➕ Thêm môn học mới';
            document.getElementById('subjectId').value = '';
            openModal('subjectModal');
        });
    }

    const addStudentBtn = document.getElementById('addStudentBtn');
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
                    : '<option value="" disabled>-- Không có Face ID mới, vui lòng train thêm --</option>';
            }
            openModal('studentModal');
        });
    }

    // =============================================================
    // === BỘ ĐIỀU HƯỚNG TẢI DỮ LIỆU BAN ĐẦU ===
    // =============================================================
    if (document.getElementById('statsGrid')) {
        // loadDashboardData(); // Có thể thêm lại hàm này nếu cần
    } else if (document.getElementById('subjectGrid')) {
        loadSubjectsData();
    } else if (document.getElementById('studentList')) {
        loadStudentsData();
    }
});