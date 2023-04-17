import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_


class Model(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out):
        super(Model, self).__init__()
        self.n_inputs=n_inputs
        self.n_hidden=n_hidden
        self.n_out=n_out

        self.h1 = nn.Linear(self.n_inputs, n_hidden)
        self.a1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(n_hidden)

        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.a2 = nn.ReLU()
        self.d2 = nn.Dropout(p=0.1)

        self.h3 = nn.Linear(n_hidden, n_out)
        self.a3 = nn.Sigmoid()

    def forward(self, inp):
        out = self.a1(self.h1(inp))
        out = self.b1(out)
        out = self.a2(self.h2(out))
        out = self.d2(out)
        out = self.a3(self.h3(out))
        return out


"""
Batch Normalization: Thêm lớp Batch Normalization giữa các lớp để chuẩn hóa giá trị đầu ra của lớp trước khi đưa vào lớp tiếp theo.
Điều này giúp giảm việc các giá trị đầu ra rơi vào khoảng giá trị phi tuyến của hàm kích hoạt, giúp cải thiện hiệu suất mô hình.

Xavier Initialization, He Initialization: Đây là phương pháp khởi tạo trọng số cho mô hình, giúp cho gradient được lan truyền tốt hơn và tránh hiện tượng gradient vanishing.
Xavier Initialization sử dụng hàm xavier_uniform_() hoặc xavier_normal_() để khởi tạo trọng số.
He Initialization, sử dụng hàm "kaiming_normal_ "

"""
class Model2(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out):
        super(Model2, self).__init__()
        self.n_inputs=n_inputs
        self.n_hidden=n_hidden
        self.n_out=n_out

        self.h1 = nn.Linear(self.n_inputs, n_hidden)
        kaiming_uniform_(self.h1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.a1 = nn.ReLU()

        self.h2 = nn.Linear(n_hidden, n_hidden)
        kaiming_uniform_(self.h2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.a2 = nn.ReLU()
        self.d2 = nn.Dropout(p=0.1)

        self.h3 = nn.Linear(n_hidden, n_out)
        xavier_uniform_(self.h3.weight)
        self.a3 = nn.Sigmoid()

    def forward(self, inp):
        out = self.a1(self.bn1(self.h1(inp)))
        out = self.a2(self.bn2(self.h2(out)))
        out = self.d2(out)
        out = self.a3(self.h3(out))
        return out
"""
sử dụng cơ chế skip connection
"""

class Skip_Model(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out):
        super(Skip_Model, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.h1 = nn.Linear(self.n_inputs, n_hidden)
        self.a1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(n_hidden)

        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.a2 = nn.ReLU()
        self.b2 = nn.BatchNorm1d(n_hidden)

        self.h3 = nn.Linear(n_hidden, n_out)
        self.a3 = nn.Sigmoid()

    def forward(self, inp):
        out = self.a1(self.h1(inp))
        out = self.b1(out)
        out = self.a2(self.h2(out))
        skip_out = out
        out = self.b2(out)
        out += skip_out
        out = self.a3(self.h3(out))
        return out


