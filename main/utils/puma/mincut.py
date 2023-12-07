import networkx as nx
import numpy as np

def mincut(sourcesink, remain):
    # Khởi tạo đồ thị có hướng
    G = nx.DiGraph()

    # Chuyển đổi ma trận từ MATLAB sang danh sách trong Python
    # sourcesink = np.array(sourcesink)
    # remain = np.array(remain)

    # Thêm cạnh và trọng số từ SOURCESINK
    for node, source_weight, sink_weight in sourcesink[:, :3]:
        G.add_edge('source', int(node), capacity=float(source_weight))
        G.add_edge(int(node), 'sink', capacity=float(sink_weight))

    # Thêm cạnh và trọng số từ REMAIN
    for start_node, end_node, direct_weight, inverse_weight in remain[:, :4]:
        G.add_edge(int(start_node), int(end_node), capacity=float(direct_weight))
        G.add_edge(int(end_node), int(start_node), capacity=float(inverse_weight))

    # Tìm luồng cực đại và cắt cực tiểu
    cut_value, partition = nx.minimum_cut(G, 'source', 'sink')
    flow = nx.maximum_flow_value(G, 'source', 'sink')

    # Lưu trữ thông tin về phân chia cắt
    cutside = np.zeros((len(sourcesink), 2), dtype=int)

    # Lặp qua tất cả các đỉnh
    for i, node in enumerate(G.nodes):
        if node not in ['source', 'sink']:
            # Gán số node vào cột đầu tiên
            cutside[i-2, 0] = node
            # Gán phía của cắt cho node tương ứng
            cutside[i-2, 1] = 0 if node in partition[0] else 1

    cutside[0][0] += 1

    return flow, cutside

    
