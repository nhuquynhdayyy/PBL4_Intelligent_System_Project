document.addEventListener('DOMContentLoaded', function () {
    const selects = document.querySelectorAll('select[data-user-id]');
    const saveButtons = document.querySelectorAll('.save-assignment-btn');

    // Kích hoạt nút "Lưu" khi người dùng thay đổi lựa chọn trong dropdown
    selects.forEach(select => {
        select.addEventListener('change', function () {
            const userId = this.dataset.userId;
            const saveBtn = document.querySelector(`.save-assignment-btn[data-user-id="${userId}"]`);
            if (saveBtn) {
                saveBtn.disabled = false;
                saveBtn.classList.remove('btn-success');
                saveBtn.classList.add('btn-primary'); // Đổi màu để báo hiệu có thay đổi
            }
        });
    });

    // Xử lý sự kiện khi nhấn nút "Lưu"
    saveButtons.forEach(button => {
        button.addEventListener('click', async function () {
            if (this.disabled) return; // Thêm dòng này để tránh click khi nút bị vô hiệu hóa

            this.disabled = true;
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang lưu...';
            
            const userId = this.dataset.userId;
            const select = document.querySelector(`select[data-user-id="${userId}"]`);
            const classId = select.value;

            try {
                const response = await fetch(`/api/users/${userId}/assign_class`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ class_id: classId }),
                });

                const result = await response.json();
                
                if (response.ok) {
                    // Thay đổi giao diện để báo thành công
                    this.classList.remove('btn-primary');
                    this.classList.add('btn-success');
                    this.innerHTML = '<i class="fas fa-check"></i> Đã lưu';
                    // Nút sẽ tự động bị vô hiệu hóa ở lần tải lại trang tiếp theo, 
                    // hoặc bạn có thể vô hiệu hóa lại sau 1-2 giây nếu muốn
                } else {
                    alert(`Lỗi: ${result.message}`);
                    this.innerHTML = '<i class="fas fa-save"></i> Lưu';
                    this.disabled = false; // Bật lại nút nếu có lỗi
                }
            } catch (error) {
                alert('Lỗi kết nối đến server.');
                this.innerHTML = '<i class="fas fa-save"></i> Lưu';
                this.disabled = false; // Bật lại nút nếu có lỗi
            }
        });
    });
});