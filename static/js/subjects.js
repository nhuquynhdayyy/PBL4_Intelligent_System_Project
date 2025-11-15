// static/js/subjects.js
document.addEventListener('DOMContentLoaded', function() {
    const subjectGrid = document.getElementById('subjectGrid');
    const subjectForm = document.getElementById('subjectForm');
    const addSubjectBtn = document.getElementById('addSubjectBtn');
    
    async function loadSubjectsData() {
        const subjects = await apiCall('/api/subjects');
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
                     <button class="btn btn-action" data-action="history" data-id="${sub.id}" data-name="${sub.name}"><i class="fa-solid fa-clock-rotate-left"></i> Lịch sử</button>
                     <button class="btn btn-action" data-action="edit-subject" data-id="${sub.id}"><i class="fa-solid fa-pen"></i></button>
                     <button class="btn btn-danger btn-action" data-action="delete-subject" data-id="${sub.id}" data-name="${sub.name}"><i class="fa-solid fa-trash"></i></button>
                </div>
            </div>
        `).join('');
    }

    // Xử lý submit form Thêm/Sửa
    if (subjectForm) {
        subjectForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const id = document.getElementById('subjectId').value;
            const data = { name: document.getElementById('subjectName').value, icon: document.getElementById('subjectIcon').value };
            const endpoint = id ? `/api/subjects/${id}` : '/api/subjects';
            const method = id ? 'PUT' : 'POST';
            
            const result = await apiCall(endpoint, method, data);
            if (result) { alert(result.message); closeModal('subjectModal'); loadSubjectsData(); }
        });
    }

    // Xử lý nút "Thêm môn học mới"
    if (addSubjectBtn) {
        addSubjectBtn.addEventListener('click', () => {
            subjectForm.reset();
            document.getElementById('subjectModalTitle').textContent = '➕ Thêm môn học mới';
            document.getElementById('subjectId').value = '';
            openModal('subjectModal');
        });
    }

    // Xử lý các nút Sửa/Xóa/Lịch sử bằng Event Delegation
    document.body.addEventListener('click', async (e) => {
        const target = e.target.closest('.btn-action[data-action]');
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
        } else if (action === 'delete-subject') {
            if (confirm(`Bạn có chắc muốn xóa môn học "${name}"?\nMọi dữ liệu liên quan (buổi học, phát biểu) cũng sẽ bị xóa!`)) {
                const result = await apiCall(`/api/subjects/${id}`, 'DELETE');
                if (result) { alert(result.message); loadSubjectsData(); }
            }
        } else if (action === 'history') {
             // ... logic hiển thị modal lịch sử ...
        }
    });

    // Tải dữ liệu lần đầu khi vào trang
    loadSubjectsData();
});