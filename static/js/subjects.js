// static/js/subjects.js
document.addEventListener('DOMContentLoaded', function() {
    const subjectGrid = document.getElementById('subjectGrid');
    const addSubjectBtn = document.getElementById('addSubjectBtn');

    // DOM elements cho modal
    const subjectForm = document.getElementById('subjectForm');
    const subjectModalTitle = document.getElementById('subjectModalTitle');
    const subjectIdInput = document.getElementById('subjectId');
    const subjectNameInput = document.getElementById('subjectName');
    const subjectIconInput = document.getElementById('subjectIcon');
    const subjectCategorySelect = document.getElementById('subjectCategory');

    // Hàm tải và hiển thị danh sách môn học
    async function loadSubjectsData() {
        const subjects = await apiCall('/api/subjects');
        if (!subjects) {
            subjectGrid.innerHTML = '<p class="error">Lỗi tải dữ liệu môn học.</p>';
            return;
        }
        
        if (subjects.length === 0) {
            subjectGrid.innerHTML = '<p>Chưa có môn học nào. Nhấn "Thêm môn học mới" để bắt đầu.</p>';
            return;
        }

        // CẬP NHẬT: Hiển thị thêm 'category'
        subjectGrid.innerHTML = subjects.map(sub => `
            <div class="card subject-card-item">
                <div class="card-body">
                    <h5 class="card-title">${sub.icon || '📚'} ${sub.name}</h5>
                    <p class="card-text"><strong>Loại:</strong> ${sub.category || 'Chưa phân loại'}</p>
                    <p class="card-text"><strong>Buổi học:</strong> ${sub.session_count}</p>
                    <p class="card-text"><strong>Phát biểu:</strong> ${sub.total_speeches}</p>
                    <div class="item-actions">
                         <button class="btn btn-secondary btn-sm" data-action="history" data-id="${sub.id}" data-name="${sub.name}">
                            <i class="fas fa-history"></i> Lịch sử
                         </button>
                         <button class="btn btn-primary btn-sm" data-action="edit-subject" data-id="${sub.id}">
                            <i class="fas fa-edit"></i> Sửa
                         </button>
                         <button class="btn btn-danger btn-sm" data-action="delete-subject" data-id="${sub.id}" data-name="${sub.name}">
                            <i class="fas fa-trash"></i> Xóa
                         </button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    // Xử lý submit form Thêm/Sửa
    if (subjectForm) {
        subjectForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const id = subjectIdInput.value;
            // CẬP NHẬT: Gửi thêm 'category'
            const data = {
                name: subjectNameInput.value,
                icon: subjectIconInput.value,
                category: subjectCategorySelect.value
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

    // Xử lý nút "Thêm môn học mới"
    if (addSubjectBtn) {
        addSubjectBtn.addEventListener('click', () => {
            subjectForm.reset();
            subjectModalTitle.textContent = '➕ Thêm môn học mới';
            subjectIdInput.value = '';
            openModal('subjectModal');
        });
    }

    // Xử lý các nút Sửa/Xóa/Lịch sử bằng Event Delegation
    subjectGrid.addEventListener('click', async (e) => {
        const target = e.target.closest('button[data-action]');
        if (!target) return;
        
        const action = target.dataset.action;
        const id = target.dataset.id;
        const name = target.dataset.name;

        if (action === 'edit-subject') {
            const subject = await apiCall(`/api/subjects/${id}`);
            if (!subject) return;
            subjectModalTitle.textContent = `✏️ Chỉnh sửa môn học`;
            subjectIdInput.value = subject.id;
            subjectNameInput.value = subject.name;
            subjectIconInput.value = subject.icon;
            // CẬP NHẬT: Điền dữ liệu 'category' vào form
            subjectCategorySelect.value = subject.category || '';
            openModal('subjectModal');
        } else if (action === 'delete-subject') {
            if (confirm(`Bạn có chắc muốn xóa môn học "${name}"?\nMọi dữ liệu liên quan (buổi học, phát biểu, điểm số) cũng sẽ bị xóa!`)) {
                const result = await apiCall(`/api/subjects/${id}`, 'DELETE');
                if (result) {
                    alert(result.message);
                    loadSubjectsData();
                }
            }
        } else if (action === 'history') {
             alert(`Chức năng xem lịch sử cho môn "${name}" sẽ được phát triển sau.`);
        }
    });

    // Tải dữ liệu lần đầu khi vào trang
    loadSubjectsData();
});
