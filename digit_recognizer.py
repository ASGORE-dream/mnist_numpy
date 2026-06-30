import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# ---------- 加载训练好的模型 ----------
w1 = np.load('w1.npy')
b1 = np.load('b1.npy')
w2 = np.load('w2.npy')
b2 = np.load('b2.npy')

# ---------- 预测函数 ----------
def predict_digit(img_28x28):
    """输入 28x28 的像素数组，返回预测结果"""
    pixels = img_28x28.reshape(1, 784) / 255.0
    
    z1 = pixels @ w1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b2
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
    
    pred = int(np.argmax(a2))
    prob = float(np.max(a2)) * 100
    return pred, prob

# ---------- GUI ----------
class DigitCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        
        self.canvas_size = 280      # 显示画板 / 高清绘图大小
        self.pen_size = 16          # 画笔粗细（像素）
        
        # 创建画板（显示用）
        self.canvas = tk.Canvas(root, width=self.canvas_size,
                                height=self.canvas_size, bg='black')
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)
        
        # 创建高清 PIL 图像（280x280），识别时再缩小到 28x28
        self.high_res_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.high_res_image)
        
        # 按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="识别", command=self.recognize,
                  width=10, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="清空", command=self.clear,
                  width=10, height=2).pack(side=tk.LEFT, padx=5)
        
        # 结果显示
        self.result_label = tk.Label(root, text="请在上面写一个数字 (0-9)",
                                     font=("Arial", 16))
        self.result_label.pack(pady=10)
        
        self.last_x = None
        self.last_y = None
    
    def paint(self, event):
        x, y = event.x, event.y
        
        # 在 tkinter 画板上画圆
        r = self.pen_size
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill='white', outline='white')
        
        # 在高清 PIL 图像上画圆（同尺寸）
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        
        # 如果上一个点存在，连成线（让笔画连续）
        if self.last_x is not None:
            # tkinter 画线
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill='white', width=r * 2, capstyle='round')
            # PIL 画线
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=255, width=r * 2)
        
        self.last_x = x
        self.last_y = y
    
    def reset_last(self, event):
        self.last_x = None
        self.last_y = None
    
    def recognize(self):
        """识别画板上的数字"""
        # 高清图 → 缩放到 28x28（PIL 自动抗锯齿）
        small_img = self.high_res_image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(small_img, dtype=np.float32)
        
        # 预测
        pred, prob = predict_digit(img_array)
        self.result_label.config(text=f"预测结果: {pred}　置信度: {prob:.1f}%")
    
    def clear(self):
        """清空画板"""
        self.canvas.delete("all")
        self.high_res_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.high_res_image)
        self.result_label.config(text="已清空，请写一个数字 (0-9)")
        self.last_x = None
        self.last_y = None

# ---------- 启动 ----------
if __name__ == '__main__':
    root = tk.Tk()
    app = DigitCanvas(root)
    root.mainloop()
