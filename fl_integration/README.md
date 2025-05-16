# Hướng dẫn Tích hợp và Sử dụng Federated Learning với Lifecycle Demo

## Tổng quan

Tài liệu này cung cấp hướng dẫn chi tiết về cách tích hợp và sử dụng các mô hình Federated Learning (FL) với các script mô phỏng vòng đời hệ thống ChainFLIP. Hệ thống tích hợp này cho phép bạn chạy các mô hình FL song song với các giai đoạn vòng đời blockchain, từ đó phân tích dữ liệu thực tế và đưa ra các dự đoán có giá trị.

## Cấu trúc Hệ thống

Hệ thống tích hợp bao gồm các thành phần chính sau:

1. **Master Orchestration Script**: `run_integrated_system.js` - Điều phối toàn bộ quy trình
2. **Lifecycle Demo Scripts**: Các script trong thư mục `SupplyChain_dapp/scripts/lifecycle_demo/`
3. **FL Integration Scripts**: Các script trong thư mục `fl_integration/`
   - `run_sybil_detection.py` - Phát hiện tấn công Sybil
   - `run_batch_monitoring.py` - Giám sát xử lý lô hàng
   - `run_node_behavior_timeseries.py` - Phân tích hành vi node theo thời gian
   - `run_arbitrator_bias.py` - Phân tích thiên vị của trọng tài
   - `run_dispute_risk.py` - Phân tích rủi ro tranh chấp
   - `validate_integration.py` - Kiểm tra tính hợp lệ của tích hợp

## Yêu cầu Hệ thống

1. **Node.js**: Phiên bản 14.0.0 trở lên
2. **Python**: Phiên bản 3.8 trở lên
3. **TensorFlow**: Phiên bản 2.5.0 trở lên
4. **TensorFlow Federated**: Phiên bản 0.20.0 trở lên
5. **Web3.py**: Phiên bản 5.23.0 trở lên
6. **Hardhat**: Phiên bản 2.9.0 trở lên

## Cài đặt

1. **Cài đặt các gói Node.js**:
   ```bash
   cd /path/to/Project
   npm install
   ```

2. **Cài đặt các gói Python**:
   ```bash
   pip install tensorflow tensorflow-federated web3 pandas numpy matplotlib
   ```

3. **Cấu hình môi trường**:
   - Đảm bảo file `.env` trong thư mục `w3storage-upload-script` đã được cấu hình đúng với thông tin RPC URL và địa chỉ contract
   - Kiểm tra đường dẫn trong `blockchain_connector.py` trỏ đến file `.env` đúng

## Hướng dẫn Sử dụng

### Chạy Toàn bộ Hệ thống Tích hợp

Để chạy toàn bộ hệ thống tích hợp từ đầu đến cuối:

```bash
cd /path/to/Project
node run_integrated_system.js
```

Script này sẽ tự động:
1. Chạy các script lifecycle_demo theo thứ tự
2. Trigger các mô hình FL tại các điểm thích hợp
3. Ghi log và lưu kết quả

### Chạy Từng Giai đoạn Riêng biệt

Nếu bạn muốn chạy từng giai đoạn một cách riêng biệt:

1. **Chạy các script lifecycle_demo**:
   ```bash
   cd /path/to/Project/SupplyChain_dapp/scripts/lifecycle_demo
   node 01_deploy_and_configure.cjs
   node 02_scenario_product_creation.cjs
   # ... và các script tiếp theo
   ```

2. **Chạy các mô hình FL**:
   ```bash
   cd /path/to/Project
   python fl_integration/run_sybil_detection.py
   python fl_integration/run_node_behavior_timeseries.py
   # ... và các mô hình khác
   ```

### Kiểm tra Tính Hợp lệ của Tích hợp

Để kiểm tra xem tất cả các thành phần đã được tích hợp đúng cách:

```bash
cd /path/to/Project
python fl_integration/validate_integration.py
```

Script này sẽ kiểm tra:
- Sự tồn tại của tất cả các script cần thiết
- Tính hợp lệ của file context
- Khả năng đọc/ghi dữ liệu của các script FL
- Luồng dữ liệu giữa các thành phần

## Luồng Dữ liệu và Context

Hệ thống sử dụng file `demo_context.json` để truyền thông tin giữa các script. File này được tạo và cập nhật bởi các script lifecycle_demo, và được đọc bởi các script FL.

Các thông tin quan trọng trong context bao gồm:
- Địa chỉ contract
- Địa chỉ các node (manufacturer, transporter, retailer, buyer, arbitrator)
- Thông tin sản phẩm (token ID, batch number, v.v.)
- Thông tin giao dịch và vận chuyển
- Thông tin tranh chấp và giải quyết

## Kết quả và Log

Các kết quả và log được lưu tại:
- **Log file**: `fl_integration_run.log` trong thư mục gốc của Project
- **Kết quả FL**: Thư mục `fl_integration/results/`
- **Context file**: `SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json`

## Khắc phục Sự cố

### Vấn đề với Context File

Nếu gặp lỗi "Context file not found":
- Đảm bảo đã chạy các script lifecycle_demo theo đúng thứ tự
- Kiểm tra đường dẫn đến file context trong các script FL

### Lỗi Import Module

Nếu gặp lỗi "No module named...":
- Đảm bảo đã cài đặt đầy đủ các gói Python cần thiết
- Kiểm tra đường dẫn trong `sys.path.append()` trong các script FL

### Lỗi Kết nối Blockchain

Nếu gặp lỗi kết nối blockchain:
- Kiểm tra thông tin RPC URL trong file `.env`
- Đảm bảo mạng blockchain (Amoy testnet hoặc mạng khác) đang hoạt động
- Kiểm tra địa chỉ contract trong file `.env` là chính xác

## Mở rộng và Tùy chỉnh

### Thêm Mô hình FL Mới

Để thêm một mô hình FL mới:
1. Tạo script mới trong thư mục `fl_integration/`
2. Sử dụng mẫu từ các script hiện có
3. Cập nhật `run_integrated_system.js` để gọi script mới tại điểm thích hợp

### Tùy chỉnh Lifecycle Demo

Nếu bạn muốn tùy chỉnh các script lifecycle_demo:
1. Chỉnh sửa các script trong thư mục `SupplyChain_dapp/scripts/lifecycle_demo/`
2. Đảm bảo vẫn cập nhật file `demo_context.json` với thông tin cần thiết
3. Kiểm tra lại tích hợp với `validate_integration.py`

## Khuyến nghị Kiểm thử Thực tế

Khi triển khai trên môi trường thực tế của bạn:

1. **Kiểm thử từng thành phần**: Chạy từng script một cách riêng biệt để đảm bảo chúng hoạt động đúng
2. **Kiểm thử tích hợp**: Chạy `validate_integration.py` để kiểm tra tính hợp lệ của tích hợp
3. **Kiểm thử end-to-end**: Chạy `run_integrated_system.js` để kiểm tra toàn bộ quy trình
4. **Kiểm thử với dữ liệu thực**: Sử dụng dữ liệu thực từ blockchain của bạn để kiểm tra các mô hình FL

## Kết luận

Hệ thống tích hợp này cho phép bạn chạy các mô hình Federated Learning song song với các giai đoạn vòng đời blockchain, từ đó phân tích dữ liệu thực tế và đưa ra các dự đoán có giá trị. Bằng cách tuân theo hướng dẫn này, bạn có thể dễ dàng triển khai và sử dụng hệ thống trên môi trường của mình.

Nếu có bất kỳ câu hỏi hoặc vấn đề nào, vui lòng tham khảo các file log và kết quả để biết thêm thông tin chi tiết.
