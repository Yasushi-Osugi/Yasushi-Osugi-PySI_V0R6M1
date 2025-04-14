#main250402.py


#PySI_V0R6M1/
#├── main.py                              ← エントリーポイント
#├── pysi/
#│   ├── gui/
#│   │   └── app.py                       ← GUI関連
#│   ├── network/
#│   │   └── tree.py                      ← ネットワーク構造
#│   ├── plan/
#│   │   ├── demand_generate.py          ← 需要生成
#│   │   └── operations.py               ← オペレーション
#│   └── utils/
#│       ├── config.py                   ← 設定ファイル読み書き
#│       └── file_io.py                  ← ファイル読み込み系
#└── _data_parameters/
#    └── ...（データと補助スクリプト）





# main.py
from pysi.utils.config import Config

from pysi.gui.app import *


import tkinter as tk

def main():
    # Create a global configuration instance
    config = Config()

    # Initialize GUI with the configuration
    root = tk.Tk()
    app = PSIPlannerApp(root, config)
    root.mainloop()

if __name__ == "__main__":
    main()


