# Yasushi-Osugi-PySI_V0R6M1
PySI named "Global Weekly PSI Planner" is python based Supply Chain Planner for visualising and evaluating  global operations. 

PySI V0R6M1機能概要


📄 main.py のコード（抜粋）

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
________________________________________
🔍 処理の流れ
1.	設定読み込み → pysi.utils.config.Config
2.	GUIの起動 → pysi.gui.app.PSIPlannerApp を tkinter.Tk() 上に構築
3.	Tk GUIのメインループ実行 → mainloop()
________________________________________

pysi.gui.app.py 
このファイルには PSIPlannerApp クラス を中心とした GUI のロジックが実装されており、主に以下のような構成になっています：
________________________________________
🎯 主な機能・構成
🧠 初期処理：
•	Config() オブジェクトを使って初期設定読み込み
•	S_month_optimized.csv を読み込み → 月次需要を週次に変換（process_monthly_demand()）
•	ツリー構造（root_node_out_opt など）を構築
•	PSI 空間（週次×4列のリスト）を各ノードに割当てる：make_psi_space_dict(), set_dict2tree_psi()
📊 PSIグラフ生成：
•	show_psi(), show_psi_graph(), show_psi_graph4opt() などで PSI のグラフを生成
•	matplotlib + FigureCanvasTkAgg で Tkinter GUI に表示
📁 データローディング：
•	フォルダ内 .csv を走査し、自動で inbound/outbound を判定して読み込み
________________________________________
 

class PSIPlannerApp:このクラスは main.py から呼び出されて、Tkinter アプリの起点として機能しています。
________________________________________
PSI 関数カテゴリ詳細
番号	関数名	カテゴリ	カテゴリ機能概要
1	eval_buffer_stock	利益・コスト分析	利益率、在庫コスト、キャッシュフローなどの経済的観点からPSIやネットワークを評価します。
2	psi_price4cf	利益・コスト分析	利益率、在庫コスト、キャッシュフローなどの経済的観点からPSIやネットワークを評価します。
3	cashflow_out_in_net	利益・コスト分析	利益率、在庫コスト、キャッシュフローなどの経済的観点からPSIやネットワークを評価します。
4	demand_planning	需給計画（流通系:outbound 需要側:demand side）	需要計画を立て生成し、ノードや週次に紐付けたPSI構造を構築します。
5	supply_planning	需給計画（流通系:outbound 供給側:supply side）	供給計画を立て生成し、ノードや週次に紐付けたPSI構造を構築します。
6	demand_leveling	需給計画（需要平準化）	需要平準化処理を行い、ノードや週次に紐付けたPSI構造を構築します。

7	show_revenue_profit	可視化・UI表示	TkinterベースのGUIとしてネットワーク構造やPSIグラフ、棒グラフ、3Dビューなどを可視化します。
8	show_cost_structure_b	可視化・UI表示	TkinterベースのGUIとしてネットワーク構造やPSIグラフ、棒グラフ、3Dビューなどを可視化します。
9	show_month_data_csv	可視化・UI表示	TkinterベースのGUIとしてネットワーク構造やPSIグラフ、棒グラフ、3Dビューなどを可視化します。
10	show_3d_overview	可視化・UI表示	ネットワーク構造やPSIグラフ、棒グラフ、3Dビューなどを可視化します。
11	supplychain_performa	データ保存・出力	PSIやネットワーク情報などをCSV形式で保存し、外部分析やレポート作成に使用します。
12	inbound_lot_by_lot_to_csv	データ保存・出力	PSIやネットワーク情報などをCSV形式で保存し、外部分析やレポート作成に使用します。
13	outbound_lot_by_lot	データ保存・出力	PSIやネットワーク情報などをCSV形式で保存し、外部分析やレポート作成に使用します。
14	outbound_psi_to_csv	データ保存・出力	PSIやネットワーク情報などをCSV形式で保存し、外部分析やレポート作成に使用します。
15	save_to_directory	データ保存・出力	保存対象をディレクトリ単位で保存します。
16	lot_cost_structure_to_csv	データ保存・出力	PSIやネットワーク情報などをCSV形式で保存し、外部分析やレポート作成に使用します。
17	inbound_psi_to_csv	データ保存・出力	PSIやネットワーク情報などをCSV形式で保存し、外部分析やレポート作成に使用します。
18	load_data_files	データ読み込み	CSVやディレクトリ構成から、PSI構造、需要データ、供給データなどを読み込みます。
19	load_from_directory	データ読み込み	CSVやディレクトリ構成から、PSI構造、需要データ、供給データなどを読み込みます。
20	optimize_network	ネットワーク最適化	最適化ソルバーなどを用い、PSIやネットワークの付帯構造や構成を最適化します。
21	on_exit	アプリ終了処理	GUIアプリケーションを終了させます。
22	Inbound_DmBw	需給計画 生産系:Inboundのdemand plan需要計画
23  Inbound_SpFw	需給計画 生産系:Inboundのsupply plan供給計画  
________________________________________


カテゴリ	含まれる関数例
💰 評価・コスト分析	eval_buffer_stock, psi_price4cf, cashflow_out_in_net
📈 計画生成（需要・供給）	demand_planning, supply_planning
📥 データ読み込み	load_data_files, load_from_directory
📤 データ保存・出力	inbound_psi_to_csv, save_to_directory, lot_cost_structure_to_csv
📊 可視化・UI表示	show_3d_overview, show_cost_stracture_bar_graph, scrollbar
🔄 ネットワーク最適化	optimize_network
🚪 アプリ終了	on_exit
📄 Excel連携	psi_for_excel

