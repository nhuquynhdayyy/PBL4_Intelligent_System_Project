document.addEventListener('DOMContentLoaded', function() {
    const editModal = new bootstrap.Modal(document.getElementById('editClassModal'));
    const editForm = document.getElementById('editClassForm');
    const editClassId = document.getElementById('editClassId');
    const editClassName = document.getElementById('editClassName');
    const editAcademicYear = document.getElementById('editAcademicYear');
    const saveChangesBtn = document.getElementById('saveChangesBtn');
    const editErrorAlert = document.getElementById('editErrorAlert');

    // Sự kiện khi nhấn nút "Sửa" trên một dòng
    document.querySelectorAll('.edit-btn').forEach(button => {
        button.addEventListener('click', async function() {
            const classId = this.dataset.id;
            
            // Gọi API để lấy dữ liệu mới nhất của lớp học
            const response = await fetch(`/api/classes/${classId}`);
            if (response.ok) {
                const data = await response.json();
                // Điền dữ liệu vào form trong modal
                editClassId.value = data.id;
                editClassName.value = data.name;
                editAcademicYear.value = data.academic_year;
                editErrorAlert.classList.add('d-none'); // Ẩn thông báo lỗi cũ
            } else {
                alert('Không thể tải dữ liệu lớp học.');
            }
        });
    });

    // Sự kiện khi nhấn nút "Lưu thay đổi" trong modal
    saveChangesBtn.addEventListener('click', async function() {
        const classId = editClassId.value;
        const data = {
            name: editClassName.value.trim(),
            academic_year: editAcademicYear.value.trim()
        };

        if (!data.name || !data.academic_year) {
            editErrorAlert.textContent = 'Vui lòng điền đầy đủ thông tin.';
            editErrorAlert.classList.remove('d-none');
            return;
        }

        const response = await fetch(`/api/classes/${classId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok) {
            editModal.hide();
            // Tải lại trang để thấy thay đổi
            // Cách tốt hơn là cập nhật DOM, nhưng reload đơn giản và hiệu quả
            window.location.reload(); 
        } else {
            // Hiển thị lỗi từ server trong modal
            editErrorAlert.textContent = result.message || 'Có lỗi xảy ra.';
            editErrorAlert.classList.remove('d-none');
        }
    });

    // Sự kiện khi nhấn nút "Xóa"
    document.querySelectorAll('.delete-btn').forEach(button => {
        button.addEventListener('click', async function() {
            const classId = this.dataset.id;
            const className = this.dataset.name;

            if (confirm(`Bạn có chắc chắn muốn xóa lớp học "${className}" không? Hành động này không thể hoàn tác.`)) {
                const response = await fetch(`/api/classes/${classId}`, {
                    method: 'DELETE'
                });

                const result = await response.json();
                
                if (response.ok) {
                    // Xóa dòng tương ứng khỏi bảng
                    document.getElementById(`class-row-${classId}`).remove();
                    // Hiển thị thông báo thành công (có thể dùng alert hoặc một thư viện toast)
                    alert(result.message);
                } else {
                    // Hiển thị thông báo lỗi từ server
                    alert(`Lỗi: ${result.message}`);
                }
            }
        });
    });
});