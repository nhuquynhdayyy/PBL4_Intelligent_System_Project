document.addEventListener('DOMContentLoaded', function () {
    const assignSelectors = document.querySelectorAll('.assign-selector');
    const saveBtns = document.querySelectorAll('.save-assign-btn');

    // --- HÀM HIỂN THỊ POP-UP TÙY CHỈNH ---
    function showPopup(message) {
        return new Promise((resolve) => {
            const modal = document.getElementById('customConfirmModal');
            const msgEl = document.getElementById('customConfirmMessage');
            const okBtn = document.getElementById('okConfirmBtn');
            const cancelBtn = document.getElementById('cancelConfirmBtn');

            msgEl.innerText = message;
            modal.classList.add('show'); // Hiển thị modal

            // Xử lý khi nhấn Đồng ý
            okBtn.onclick = function() {
                modal.classList.remove('show');
                resolve(true);
            };

            // Xử lý khi nhấn Hủy
            cancelBtn.onclick = function() {
                modal.classList.remove('show');
                resolve(false);
            };
        });
    }

    // --- HÀM GỬI DỮ LIỆU ---
    async function performAssignment(classId, teacherId, confirmSwitch = false) {
        const btn = document.querySelector(`.save-assign-btn[data-class-id="${classId}"]`);

        try {
            const res = await fetch(`/api/classes/${classId}/assign_teacher`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    teacher_id: teacherId,
                    confirm_switch: confirmSwitch 
                })
            });

            const result = await res.json();

            // Nếu gặp xung đột (Mã 409) -> Hiện Pop-up tùy chỉnh
            if (res.status === 409) {
                const confirmed = await showPopup(result.message);
                if (confirmed) {
                    await performAssignment(classId, teacherId, true);
                } else {
                    location.reload();
                }
                return;
            }

            if (res.ok) {
                alert("Cập nhật phân công thành công!");
                location.reload();
            } else {
                alert("Lỗi: " + result.message);
                location.reload();
            }
        } catch (err) {
            alert("Lỗi kết nối máy chủ!");
            location.reload();
        }
    }

    // Sự kiện nhấn nút Lưu
    saveBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            const classId = this.dataset.classId;
            const teacherId = document.querySelector(`.assign-selector[data-class-id="${classId}"]`).value;

            this.disabled = true;
            this.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

            performAssignment(classId, teacherId, false);
        });
    });

    // Bật nút Lưu khi đổi select
    assignSelectors.forEach(select => {
        select.addEventListener('change', function () {
            const classId = this.dataset.classId;
            const btn = document.querySelector(`.save-assign-btn[data-class-id="${classId}"]`);
            btn.disabled = false;
            btn.classList.replace('btn-success', 'btn-primary');
        });
    });

    // --- XỬ LÝ XÓA (CŨNG CÓ THỂ DÙNG POP-UP NÀY) ---
    const deleteBtns = document.querySelectorAll('.delete-teacher-btn');
    deleteBtns.forEach(btn => {
        btn.addEventListener('click', async function () {
            const id = this.dataset.id;
            const name = this.dataset.name;
            const confirmed = await showPopup(`Bạn có chắc chắn muốn xóa giáo viên "${name}"?`);
            if (confirmed) {
                const res = await fetch(`/api/users/${id}`, { method: 'DELETE' });
                if (res.ok) location.reload();
            }
        });
    });
});