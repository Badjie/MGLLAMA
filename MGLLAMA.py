import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import os
from transformers import LlamaModel, LlamaConfig
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns

class MGLLAMA(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(MGLLAMA, self).__init__()
        self.hidden_dim = hidden_dim

        # Replace GATConv with SAGEConv
        self.sage_input = SAGEConv(input_dim, hidden_dim)
        self.sage_hidden = SAGEConv(hidden_dim, hidden_dim)

        # LLaMA configuration
        self.llama_config = LlamaConfig(
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=0,
            vocab_size=32000  # arbitrary since we're using inputs_embeds
        )
        self.llama = LlamaModel(self.llama_config)

        self.linear_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Missing info predictors
        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing_info(self, h_v, h_N_v):
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        beta = torch.tanh(self.W_beta1(h_v) + self.W_beta2(h_N_v))
        r = torch.zeros_like(h_v)
        r_v = (gamma + 1) * r + beta
        m_v = h_v + r_v - h_N_v
        return m_v

    def forward(self, x, edge_index, h_c, batch=None):
        x_gnn = self.sage_input(x, edge_index)
        x_gnn = self.dropout(x_gnn)
        h_N_v = self.sage_hidden(x_gnn, edge_index)
        x_gnn = self.predict_missing_info(x_gnn, h_N_v)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        padded_x = torch.zeros((num_graphs, max_nodes, self.hidden_dim), device=x.device)

        for i in range(num_graphs):
            node_indices = (batch == i).nonzero(as_tuple=True)[0]
            padded_x[i, :len(node_indices), :] = x_gnn[node_indices]

        llama_output = self.llama(inputs_embeds=padded_x).last_hidden_state
        x_llama_flat = torch.cat([llama_output[i, :torch.sum(batch == i)] for i in range(num_graphs)], dim=0)

        if x_llama_flat.shape[0] != x_gnn.shape[0]:
            raise ValueError(f"Shape mismatch: x_gnn={x_gnn.shape}, x_llama={x_llama_flat.shape}")

        combined = torch.cat([x_gnn, x_llama_flat], dim=-1)
        output = self.linear_combine(combined)
        output = self.dropout(output)

        if h_c is None:
            h = output
            c = torch.zeros_like(h)
        else:
            h, c = h_c
            h = output

        return h, c


class EdgeClassifier(nn.Module):
    def __init__(self, node_embedding_dim):
        super(EdgeClassifier, self).__init__()
        edge_embedding_dim = 2 * node_embedding_dim

        #Two layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim, edge_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_embedding_dim // 2, 1),
            nn.Sigmoid()
            )
    def forward(self, edge_embeddings):
        return self.mlp(edge_embeddings)
                        
class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EnhancedTemporalGraphNetwork, self).__init__()
        self.mgllama = MGLLAMA(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.edge_classifier = EdgeClassifier(hidden_dim)  # Use the new EdgeClassifier

    def create_edge_embeddings(self, node_embeddings, edge_index):
        # If node_embeddings is a tuple, extract the first element (hidden state)
        if isinstance(node_embeddings, tuple):
            node_embeddings = node_embeddings[0]
    
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
    
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return edge_embeddings

    def forward(self, x, edge_index, batch=None):
        h = None
    
        # Get final node embeddings from mgllama
        for _ in range(self.num_layers):
            h, c = self.mgllama(x, edge_index, h, batch)
            h = (h, c)
    
        # Create edge embeddings by concatenating node embeddings
        edge_embeddings = self.create_edge_embeddings(h[0], edge_index)  # Pass h[0] instead of h
    
        # Get classification probabilities
        edge_predictions = self.edge_classifier(edge_embeddings)
        return edge_predictions, h
        
        
def create_edge_labels(G, labels, edge_index, node_to_idx):
    edge_labels = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()
        
        # Convert indices back to original node IDs
        src_node = list(G.nodes())[src_idx]
        dst_node = list(G.nodes())[dst_idx]
        
        # Create edge label based on source and destination node labels
        src_label = 1 if labels[src_node][0] == 'link_attractiveness_normal' else 0
        dst_label = 1 if labels[dst_node][0] == 'link_attractiveness_normal' else 0
        
        # Edge is labeled as normal if either source or destination is normal
        edge_labels.append(float(src_label or dst_label))
    
    return torch.tensor(edge_labels, dtype=torch.float)

def weighted_cross_entropy_loss(predictions, targets, pos_weight):
    """
    Custom weighted cross entropy loss
    N1: number of positive samples
    N2: number of negative samples
    """
    epsilon = 1e-7  # Small constant to prevent log(0)
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    
    # Calculate weighted loss
    loss = -(pos_weight * targets * torch.log(predictions) + 
             (1 - targets) * torch.log(1 - predictions))
    
    return loss.mean()

def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['timestamp'])
    return G

def calculate_link_and_antilink_scores(G):
    link_scores = nx.out_degree_centrality(G)
    antilink_scores = nx.eigenvector_centrality(G, max_iter=1000)
    return link_scores, antilink_scores

def label_nodes(link_scores, antilink_scores, link_threshold=0.01, antilink_threshold=0.01):
    labels = {}
    for node in link_scores:
        link_attractiveness_label = 'link_attractiveness_normal' if link_scores[node] > link_threshold else 'link_attractiveness_abnormal'
        link_affinity_label = 'link_affinity_normal' if antilink_scores[node] > antilink_threshold else 'link_affinity_abnormal'
        labels[node] = (link_attractiveness_label, link_affinity_label)
    return labels

def label_edges(G):
    ego_networks = defaultdict(nx.DiGraph)
    for u, v, data in G.edges(data=True):
        ego_networks[u].add_edge(u, v, weight=data['weight'])
        ego_networks[v].add_edge(u, v, weight=data['weight'])
    return ego_networks

def count_edges(ego_network, label):
    count = 0
    for u, v, data in ego_network.edges(data=True):
        if label in data:
            count += 1
    return count

def common_eval(ego_networks):
    neighbors = {}
    for node, ego_net in ego_networks.items():
        neighbors[node] = list(ego_net.neighbors(node))
    return neighbors

def extract_features(G, node):
    ego_networks = label_edges(G)
    neighbors = common_eval(ego_networks)
    P1 = count_edges(ego_networks[node], label='link_attractiveness_normal')
    P2 = count_edges(ego_networks[node], label='link_attractiveness_abnormal')
    P3 = count_edges(ego_networks[node], label='link_affinity_normal')
    P4 = count_edges(ego_networks[node], label='link_affinity_abnormal')
    
    node_features = [P1, P2, len(neighbors.get(node, [])), 
                    P3, P4, len(neighbors.get(node, [])), 
                    G.in_degree(node), G.out_degree(node)]
    return node_features

def create_data_list(G_list, labels_list):
    data_list = []
    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
        
        # Create edge index for the entire graph
        edge_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) 
                                  for u, v in G.edges], dtype=torch.long).t().contiguous()
        
        # Create feature matrix for all nodes
        x = torch.tensor([extract_features(G, node) for node in G.nodes], 
                        dtype=torch.float)
        
        # Create edge labels
        edge_labels = create_edge_labels(G, labels, edge_index, node_to_idx)
        
        # Create a single Data object for the entire graph
        data = Data(x=x, edge_index=edge_index, y=edge_labels)
        data_list.append(data)
    
    return data_list


    
def train_model(model, train_loader, epochs=10, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        for data in train_loader:
            optimizer.zero_grad()
            edge_predictions, _ = model(data.x, data.edge_index, data.batch)
            loss = weighted_cross_entropy_loss(edge_predictions, data.y, pos_weight=1.0)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            edge_predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            all_predictions.append(edge_predictions.squeeze().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    binary_preds = (all_predictions > 0.5).astype(int)

    # Compute metrics
    auc_score = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)

  
    # Confusion Matrix
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal_Link", "Abnormal_Link"], yticklabels=["Normal_Link", "Abnormal_Link"])
    plt.xlabel("Predicted", fontsize=22)
    plt.ylabel("True", fontsize=22)
    plt.title("Confusion Matrix of BTA Dataset", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    # AUC-ROC Curve of BTA
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate", fontsize=22)
    plt.ylabel("True Positive Rate", fontsize=22)
    plt.title("ROC Curve", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Precision-Recall Curve of BTA
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='purple')
    plt.xlabel("Recall", fontsize=22)
    plt.ylabel("Precision", fontsize=22)
    plt.title("Precision-Recall Curve", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()

    return auc_score, precision, recall, f1

def main():
    
    # Load the dataset
    #file_path = 'token_transfers.csv'
    file_path = 'E:/ansu/ERC20-stablecoins/soc-sign-bitcoinalpha.csv'
    data = pd.read_csv(file_path)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Sort data by timestamp
    data = data.sort_values(by='timestamp')

    # Split data into 31-day time slices
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31

    # Combine the last few sparse time slices into a single time slice
    #threshold = 5  # Combine slices with fewer than 20 entries
    combined_time_slice = max(data['time_slice']) - 1
    data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    # Verify the new distribution of entries across time slices
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()

    # Display the new distribution
    print(new_time_slice_counts)

    
    # Create a list of graphs and labels for each time slice
    G_list = []
    labels_list = []
    
    for time_slice in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == time_slice]
        G = create_graph(slice_data)
        link_scores, antilink_scores = calculate_link_and_antilink_scores(G)
        labels = label_nodes(link_scores, antilink_scores)
        G_list.append(G)
        labels_list.append(labels)
    
    # Check if we have enough data slices for splitting
    if len(G_list) > 2:
        # Split data into train, validation, and test sets (60%, 20%, 20%)
        train_G, temp_G, train_labels, temp_labels = train_test_split(G_list, labels_list, test_size=0.4, shuffle=False)
        val_G, test_G, val_labels, test_labels = train_test_split(temp_G, temp_labels, test_size=0.5, shuffle=False)
    else:
        # If there's not enough time slices, use all data for training and skip validation/testing
        train_G, train_labels = G_list, labels_list
        val_G, val_labels, test_G, test_labels = [], [], [], []
    
    # Create dataset and dataloader
    train_data_list = create_data_list(train_G, train_labels)
    val_data_list = create_data_list(val_G, val_labels) if val_G else []
    test_data_list = create_data_list(test_G, test_labels) if test_G else []
    
    train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False) if val_data_list else None
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False) if test_data_list else None
    
    # Initialize and train model
    model = EnhancedTemporalGraphNetwork(
        input_dim=8,
        hidden_dim=16,
        num_layers=2
    )
    
    # Train the model
    trained_model = train_model(model, train_loader)
    
    # Evaluate the model on the test set if it exists
    if test_loader:
        auc_score, precision, recall, f1 = evaluate_model(trained_model, test_loader)
        print(f"Test AUC: {auc_score:.4f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
    else:
        print("Not enough data to create a test set.")
    
    return trained_model

if __name__ == "__main__":
    trained_model = main()            
