// static/js/subjects.js
document.addEventListener('DOMContentLoaded', function() {
    const subjectGrid = document.getElementById('subjectGrid');
    const addSubjectBtn = document.getElementById('addSubjectBtn');

    // DOM elements cho các Modal (đã có trong layout.html)
    const subjectForm = document.getElementById('subjectForm');
    const subjectModalTitle = document.getElementById('subjectModalTitle');
    const subjectIdInput = document.getElementById('subjectId');
    const subjectNameInput = document.getElementById('subjectName');
    const subjectIconInput = document.getElementById('subjectIcon');
    const subjectCategorySelect = document.getElementById('subjectCategory');

    // --- 1. HÀM TẢI VÀ HIỂN THỊ DANH SÁCH MÔN HỌC ---
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

    // --- 2. XỬ LÝ SUBMIT FORM (THÊM HOẶC SỬA MÔN HỌC) ---
    if (subjectForm) {
        subjectForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const id = subjectIdInput.value;
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

    // --- 3. XỬ LÝ NÚT MỞ MODAL "THÊM MỚI" ---
    if (addSubjectBtn) {
        addSubjectBtn.addEventListener('click', () => {
            subjectForm.reset();
            subjectModalTitle.textContent = '➕ Thêm môn học mới';
            subjectIdInput.value = '';
            openModal('subjectModal');
        });
    }

    // --- 4. SỬ DỤNG EVENT DELEGATION CHO CÁC NÚT TRÊN CARD ---
    subjectGrid.addEventListener('click', async (e) => {
        const target = e.target.closest('button[data-action]');
        if (!target) return;
        
        const action = target.dataset.action;
        const id = target.dataset.id;
        const name = target.dataset.name;

        // --- HÀNH ĐỘNG: SỬA MÔN HỌC ---
        if (action === 'edit-subject') {
            const subject = await apiCall(`/api/subjects/${id}`);
            if (!subject) return;
            subjectModalTitle.textContent = `✏️ Chỉnh sửa môn học`;
            subjectIdInput.value = subject.id;
            subjectNameInput.value = subject.name;
            subjectIconInput.value = subject.icon;
            subjectCategorySelect.value = subject.category || '';
            openModal('subjectModal');
        } 
        
        // --- HÀNH ĐỘNG: XÓA MÔN HỌC ---
        else if (action === 'delete-subject') {
            if (confirm(`Bạn có chắc muốn xóa môn học "${name}"?\nMọi dữ liệu liên quan (buổi học, phát biểu, điểm số) cũng sẽ bị xóa!`)) {
                const result = await apiCall(`/api/subjects/${id}`, 'DELETE');
                if (result) {
                    alert(result.message);
                    loadSubjectsData();
                }
            }
        } 
        
       else if (action === 'history') {
            loadSessionHistory(id, name);
        }

        // --- HÀM PHỤ: TẢI LỊCH SỬ BUỔI HỌC ---
        async function loadSessionHistory(subjectId, subjectName) {
            const historyBody = document.getElementById('historyModalBody');
            const historyTitle = document.getElementById('historyModalTitle');

            historyTitle.textContent = `📊 Lịch sử buổi học - Môn ${subjectName}`;
            historyBody.innerHTML = `<div class="text-center p-4"><div class="spinner-border text-primary"></div></div>`;
            
            openModal('historyModal');

            const historyData = await apiCall(`/api/subjects/${subjectId}/history`);

            if (historyData && historyData.length > 0) {
                historyBody.innerHTML = `
                    <div class="table-responsive">
                        <table class="table table-hover mt-2">
                            <thead class="table-light">
                                <tr>
                                    <th>Buổi số</th>
                                    <th>Thời gian kết thúc</th>
                                    <th>Tổng phát biểu</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                ${historyData.map((h, index) => {
                                    // Tìm ID thực của session (Backend cần trả về ID này, nếu chưa có hãy check lại API history)
                                    // Ở đây tạm dùng h.session_id nếu bạn đã cập nhật app.py ở bước trước
                                    return `
                                    <tr class="session-row" style="cursor:pointer" data-session-id="${h.id || h.session_number}" data-num="${h.session_number}">
                                        <td><strong>#${h.session_number}</strong></td>
                                        <td>${h.end_time}</td>
                                        <td><span class="badge bg-primary">${h.speech_count} lần</span></td>
                                        <td><i class="fas fa-chevron-right text-muted"></i></td>
                                    </tr>`;
                                }).join('')}
                            </tbody>
                        </table>
                        <p class="text-muted small text-center mt-2"><i class="fas fa-info-circle"></i> Bấm vào một buổi để xem chi tiết học sinh</p>
                    </div>
                `;

                // Gán sự kiện click cho từng dòng
                document.querySelectorAll('.session-row').forEach(row => {
                    row.addEventListener('click', () => {
                        const sId = row.dataset.sessionId;
                        const sNum = row.dataset.num;
                        showSessionDetails(sId, sNum, subjectId, subjectName);
                    });
                });
            } else {
                historyBody.innerHTML = `<div class="text-center p-5"><p>Chưa có dữ liệu.</p></div>`;
            }
        }

        // --- HÀM PHỤ: XEM CHI TIẾT AI PHÁT BIỂU TRONG BUỔI ---
        async function showSessionDetails(sessionId, sessionNum, subjectId, subjectName) {
            const historyBody = document.getElementById('historyModalBody');
            const historyTitle = document.getElementById('historyModalTitle');

            historyTitle.textContent = `👥 Chi tiết buổi #${sessionNum} - Môn ${subjectName}`;
            historyBody.innerHTML = `<div class="text-center p-4"><div class="spinner-border text-primary"></div></div>`;

            const details = await apiCall(`/api/sessions/${sessionId}/details`);

            let content = `
                <button class="btn btn-sm btn-outline-secondary mb-3" id="backToHistory">
                    <i class="fas fa-arrow-left"></i> Quay lại danh sách
                </button>
            `;

            if (details && details.length > 0) {
                content += `
                    <ul class="list-group list-group-flush">
                        ${details.map(d => `
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <span><i class="fas fa-user-graduate me-2"></i> ${d.name}</span>
                                <span class="badge bg-success rounded-pill">${d.count} lần phát biểu</span>
                            </li>
                        `).join('')}
                    </ul>
                `;
            } else {
                content += `<div class="alert alert-light text-center">Không có dữ liệu phát biểu trong buổi học này.</div>`;
            }

            historyBody.innerHTML = content;

            // Nút quay lại
            document.getElementById('backToHistory').addEventListener('click', () => {
                loadSessionHistory(subjectId, subjectName);
            });
        }
    });

    // --- KHỞI CHẠY LẦN ĐẦU ---
    loadSubjectsData();
});