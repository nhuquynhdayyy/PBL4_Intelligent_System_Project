// static/js/shared.js

// Hàm mở Modal
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.add('show');
        // Ngăn cuộn trang web khi đang mở modal
        document.body.style.overflow = 'hidden';
    }
}

// Hàm đóng Modal
function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.remove('show');
        // Cho phép cuộn trang lại bình thường
        document.body.style.overflow = 'auto';
    }
}

// TỰ ĐỘNG GÁN SỰ KIỆN ĐÓNG CHO TẤT CẢ MODAL
document.addEventListener('click', function(e) {
    // 1. Nếu bấm vào nút có class 'modal-close' (nút X)
    // 2. Hoặc bấm vào nút có class 'btn-danger' hoặc 'btn-secondary' bên trong modal (nút Hủy)
    // 3. Hoặc bấm vào vùng xám bên ngoài modal
    
    if (e.target.classList.contains('modal-close') || 
        e.target.closest('.modal-close') || 
        e.target.classList.contains('btn-close-modal') ||
        (e.target.classList.contains('btn-danger') && e.target.closest('.modal')) ||
        e.target.classList.contains('modal')) 
    {
        const modal = e.target.closest('.modal');
        if (modal) {
            closeModal(modal.id);
        }
    }
});

// Hàm gọi API dùng chung
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : null
    };
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.message || `Lỗi ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error("API Error:", error);
        alert(error.message);
        return null;
    }
}