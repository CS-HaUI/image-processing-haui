# image-processing-haui
Bài tập lớn môn xử lý ảnh số và thị giác máy tính


# Đề tài [Tìm hiểu, cài đặt, ứng dụng, của phương pháp trích chọn đặc trưng LBP]

# Sinh viên thực hiện

| Tên                   | Link Github                                        |
| -----------           | -----------                                        |
| Nguyễn Văn Vũ         | [nguyenvanvutlv](https://github.com/nguyenvanvutlv)|
| Ngô Thế Tài           | [thetai](https://github.com/ngothetai)             |
| Đỗ Trung Hiếu         | [HieuDNS](https://github.com/HieuDNS)        |
| Trần Thị Khánh Linh   | [linhlukar](https://github.com/linhlukar)          |

Cấu trúc thư mục

- Cần cài đặt thư mục data ở máy local

- Và các file ảnh được đặt tên như mô tả bên dưới
    
    bắt đầu từ 1 -> N

<pre>
📦image-processing-haui
 ┣ 📂data
 ┃ ┣ 📂json
 ┃ ┃ ┣ 📜data.json
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📜1.jpg
 ┃ ┃ ┣ 📜....
 ┃ ┃ ┗ 📜N.jpg
 ┣ 📂process
 ┃ ┗ 📜function.py
 ┣ 📂visuallization
 ┃ ┣ 📂demo
 ┃ ┃ ┗ 📜result.png
 ┃ ┣ 📜plotly_custom.py
 ┃ ┗ 📜subplots.py
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜main.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜requirements.yml
</pre>

# Cách sử dụng

## Cài đặt môi trường

- Hãy chắc chắc chắn để cấu trúc như ở phần mô tả trên

- Cài đặt sẵn [Anaconda](https://www.anaconda.com/download), [gitbash](https://git-scm.com/downloads)

- Mở anaconda

``` bash
# clone repo về máy
git clone https://github.com/CS-HaUI/image-processing-haui.git
cd image-processing-haui

# chuyển sang nhánh 1-phase-1
git checkout 1-phase-1

# cài đặt môi trường có thể rất lâu khi chạy trên window
# conda env create -n ENVNAME --file ENV.yml
conda env create -n image --file requirements.yml
```

## Cách sử dụng 


``` bash
# di chuyển đến thư mục repo
cd image-processing-haui

# activate môi trường
conda activate image
```

### Đọc dữ liệu, là load dữ liệu

- file main.py

```python
# load dữ liệu từ file raw, chọn kích cỡ là (512, 512), đuôi file chọn tất cả "*"
from process.function import LBP
objects = LBP(path= "data/raw/", size= (512, 512))

"""
output:
LOADING FILE!!!
FILE: *
Read file  .*: 100%|████████████████████████████████████████████████| 32/32 [00:00<00:00, 42.14it/s]
Extract Feature: 32it [00:00, 236.76it/s]
"""
```

```python
# load dữ liệu từ file raw, chọn kích cỡ là (512, 512), đuôi file chọn tất cả "*", là lưu lại file json
from process.function import LBP
objects = LBP(path= "data/raw/", size= (512, 512))
objects.save(path_file= "data/json/data.json")

"""
output:
LOADING FILE!!!
FILE: *
Read file  .*: 100%|████████████████████████████████████████████████| 32/32 [00:00<00:00, 42.46it/s]
Extract Feature: 32it [00:00, 339.85it/s]
Save file json: 32it [00:00, 13339.07it/s]
"""
```


```python
# load dữ liệu từ file JSON
from process.function import LBP
objects = LBP(from_path= True, load_path= "data/json/data.json")
"""
output:
Read data from json: 32it [00:00, 31025.83it/s]
"""
```

### Lấy thông tin ảnh
```python
from process.function import LBP
objects = LBP(from_path= True, load_path= "data/json/data.json")

# lấy ảnh RGB từ đường dẫn
rgb_path = objects.get_image_color_from_path(path= ["data/raw/1.jpg", "data/raw/10.jpg"])
# (2, 512, 512, 3) 2 ảnh có shape là (512, 512, 3)
# nếu muốn lấy từng ảnh thì chỉ cần rgb_path[1]


# lấy ảnh RGB từ chỉ số
rgb_path = objects.get_image_color_from_id(ids= [1, 12])
# (2, 512, 512, 3) 2 ảnh có shape là (512, 512, 3)
# nếu muốn lấy từng ảnh thì chỉ cần rgb_path[1]

# lấy ảnh LBP từ đường dẫn
lbp_path = objects.get_image_lbp_from_path(path= ["data/raw/1.jpg", "data/raw/10.jpg"])
# (2, 512, 512, 3) 2 ảnh có shape là (512, 512, 3)
# nếu muốn lấy từng ảnh thì chỉ cần rgb_path[1]


# lấy ảnh RGB từ chỉ số
lbp_path = objects.get_image_lbp_from_id(ids= [1, 12])
# (2, 512, 512, 3) 2 ảnh có shape là (512, 512, 3)
# nếu muốn lấy từng ảnh thì chỉ cần rgb_path[1]
```