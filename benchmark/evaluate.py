import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_items,
        feat_dim,
        dropout=0.1
    ):
        """
        Initialize the RecSysGNN model.

        Args:
        - latent_dim (int): The dimensionality of the latent space for embeddings.
        - num_layers (int): The number of LGConv layers.
        - num_users (int): The total number of users.
        - num_items (int): The total number of items.
        - feat_dim (int): The dimensionality of additional features.
        - model: The type of model.
        - dropout (float, optional): Dropout probability (default: 0.1).
        """
        super(RecSysGNN, self).__init__()

        self.embedding = nn.Embedding(num_users + num_items, latent_dim)
        self.feat_embedding = nn.Linear(feat_dim, latent_dim)

        # Create a list of LGConv layers for message passing
        self.convs = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self):
        """
        Initialize model parameters.
        """
        # Initialize weights of the embedding layer with a normal distribution
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index, feats_tensor):
        """
        Forward pass of the RecSysGNN model.

        Args:
        - edge_index: Edge index tensor.
        - feats_tensor: Tensor containing additional features.

        Returns:
        - out: Final node embeddings after message passing.
        """
        emb = self.embedding.weight

        # Obtain embeddings for user and item features
        feats = self.feat_embedding(feats_tensor)

        # Add additional feature embeddings to the initial node embeddings
        emb = emb + feats
        embs = [emb]

        # Perform message passing through LGConv layers
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        # Aggregate node embeddings to get the final output
        out = torch.mean(torch.stack(embs, dim=0), dim=0)

        return out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index, feats_tensor):
        """
        Encode a minibatch of users, positive items, and negative items.

        Args:
        - users: Tensor of user indices.
        - pos_items: Tensor of positive item indices.
        - neg_items: Tensor of negative item indices.
        - edge_index: Edge index tensor.
        - feats_tensor: Tensor containing additional features.

        Returns:
        - Embeddings of users, positive items, and negative items.
        """
        out = self(edge_index, feats_tensor)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
        )


def get_metrics(train_df, user_Embed_wts, item_Embed_wts, n_users, n_items, test_data, K):
    """
    Compute evaluation metrics for recommendation system.

    Args:
    - user_Embed_wts: Embedding weights for users.
    - item_Embed_wts: Embedding weights for items.
    - n_users: Number of unique users.
    - n_items: Number of unique items.
    - test_data: Test dataset.
    - K: Number of top recommendations.

    Returns:
    - recall: Mean recall value.
    - precision: Mean precision value.
    """
    # Get unique user IDs from test data
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())

    # Compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    # Create dense tensor of all user-item interactions from the training data
    i = torch.stack((
        torch.LongTensor(train_df['user_id_idx'].values),
        torch.LongTensor(train_df['item_id_idx'].values)
    ))
    v = torch.ones((len(train_df)), dtype=torch.float64)
    interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items)).to_dense().to(device)

    # Mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # Compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),
                                             columns=['top_indx_' + str(x + 1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[
        ['top_indx_' + str(x + 1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id_idx',
                          right_on=['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id_idx,
                                                                               metrics_df.top_rlvnt_itm)]

    # Calculate recall and precision metrics
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean()

def get_data_from_ratings_df(dataframe):
    """
    Extracts necessary data from the ratings DataFrame to construct an edge index tensor for training.

    Args:
    - dataframe (pandas.DataFrame): DataFrame containing user-item interactions and their indices.
    - device: The device (e.g., CPU or GPU) where the tensors will reside.

    Returns:
    - n_users (int): Number of unique users present in the ratings DataFrame.
    - n_items (int): Number of unique items present in the ratings DataFrame.
    - train_edge_index (torch.Tensor): Edge index tensor representing user-item interactions for training.

    The function extracts the number of unique users and items, converts user and item indices into PyTorch LongTensors,
    and constructs an edge index tensor for training data. This tensor represents connections between users and items
    in an undirected bipartite graph, essential for training recommendation system models.
    """
    # Calculate the number of unique users from the 'user_id_idx' column in the DataFrame
    n_users = dataframe['user_id_idx'].nunique()

    # Calculate the number of unique item indices ('item_id_idx') in the DataFrame
    n_items = dataframe['item_id_idx'].nunique()

    # Convert user indices and item indices into PyTorch LongTensors
    u_t = torch.LongTensor(dataframe.user_id_idx)  # Create a LongTensor for user indices
    i_t = torch.LongTensor(dataframe.item_id_idx) + n_users  # Create a LongTensor for item indices

    # Construct edge index tensor for training data
    train_edge_index = torch.stack((
        torch.cat([u_t, i_t]),  # Concatenate user indices and item indices
        torch.cat([i_t, u_t])  # Concatenate item indices and user indices (reversed)
    )).to(device)  # Move the tensor to the specified device (e.g., CPU or GPU)

    return n_users, n_items, train_edge_index


def read_data(dir_path):
    """
    Reads and loads necessary data from CSV files.

    Returns:
    - ratings_df (pandas.DataFrame): DataFrame containing user-item interactions.
    - features_df (pandas.DataFrame): DataFrame containing node features including user and item features.
    - test_df (pandas.DataFrame): Test dataset containing user-item interactions for evaluation.

    This function reads and loads necessary data from CSV files ('ratings_df.csv', 'features_df.csv', 'test_df.csv').
    It returns three dataframes: ratings_df containing user-item interactions, features_df containing node features
    including user and item features, and test_df containing user-item interactions for evaluation purposes.
    """
    ratings_df = pd.read_csv(dir_path + '/ratings_df.csv')  # Load user-item interactions
    features_df = pd.read_csv(dir_path + '/features_df.csv')  # Load node features
    test_df = pd.read_csv(dir_path + '/test_df.csv')  # Load test dataset for evaluation

    return ratings_df, features_df, test_df


def evaluate(model, ratings_df, features_df, test_df, K):
    """
    Evaluates the recommendation model's performance using provided test data.

    Args:
    - model (torch.nn.Module): The recommendation model to be evaluated.
    - ratings_df (pandas.DataFrame): DataFrame containing user-item interactions and their indices for training.
    - features_df (pandas.DataFrame): DataFrame containing node features including user and item features.
    - test_df (pandas.DataFrame): Test dataset containing user-item interactions for evaluation.

    Returns:
    - test_topK_recall (float): Recall value indicating the model's performance in capturing relevant items in the top-K recommendations.
    - test_topK_precision (float): Precision value indicating the model's accuracy in recommending relevant items among the top-K recommendations.

    This function evaluates the recommendation model's performance using provided dataframes for training and test purposes.
    It converts the features dataframe into PyTorch tensors, extracts necessary data for constructing an edge index tensor for training,
    and obtains model outputs for evaluation. The model's outputs are used to compute recall and precision metrics based on the provided
    test dataset, returning the calculated recall and precision values as evaluation results.
    """
    # Convert features dataframe to a PyTorch tensor
    features_tensor = torch.Tensor(features_df.drop(['node_idx'], axis=1).to_numpy()).to(device)

    # Extract necessary data to construct the edge index tensor for training
    n_users, n_items, train_edge_index = get_data_from_ratings_df(ratings_df)

    model.eval()
    with torch.no_grad():
        # Obtain model outputs for evaluation
        out = model(train_edge_index, features_tensor)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))

        # Calculate evaluation metrics (recall and precision)
        test_topK_recall, test_topK_precision = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items,
                                                            test_df, K)
    return round(test_topK_recall, 4), round(test_topK_precision, 4)



def get_recommendations(model, user_id, K, ratings_df, features_df):
    """
    Generates top-K recommendations for a given user using the provided recommendation model.

    Args:
    - model (torch.nn.Module): The recommendation model used for generating predictions.
    - ratings_df (pandas.DataFrame): DataFrame containing user-item interactions and their indices.
    - features_df (pandas.DataFrame): DataFrame containing node features including user and item features.
    - user_id (int): The ID of the user for whom recommendations are requested.
    - K (int): The number of top recommendations to generate.

    Returns:
    - recommendations (list): List of the top-K recommended movie indices for the specified user.

    This function utilizes the recommendation model to generate top-K movie recommendations for a given user.
    It processes the input dataframes to obtain necessary information such as user features and constructs an
    edge index tensor for training purposes. The model predicts user-item interactions and computes scores
    for all user-item pairs. Finally, it retrieves the top-K movie indices based on the computed scores
    and returns them as a list of recommendations.
    """
    # Convert features dataframe to a PyTorch tensor
    features_tensor = torch.Tensor(features_df.drop(['node_idx'], axis=1).to_numpy()).to(device)

    # Extract necessary data to construct the edge index tensor for training
    n_users, n_items, train_edge_index = get_data_from_ratings_df(ratings_df)

    with torch.no_grad():
        # Obtain model outputs for evaluation
        out = model(train_edge_index, features_tensor)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))

        # Compute scores for all user-item pairs
        user_scores = final_user_Embed[user_id] @ final_item_Embed.t()

        # Get the top K movie indexes
        recommendations = torch.topk(user_scores, K).indices.cpu().numpy().tolist()

        return recommendations


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("output_type", type=str)
    parser.add_argument("dir_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument('--K', default=20, type=int)
    parser.add_argument('--user_id', default=16, type=int)
    args = parser.parse_args()

    ratings_df, features_df, test_df = read_data(args.dir_path)

    model = torch.load(args.model_path + '/lightgcn.pth')
    if args.output_type == 'evaluate':
        evaluate(model, ratings_df, features_df, test_df, args.K)
    elif args.model_name == 'recommend':
        get_recommendations(model, args.user_id, args.K, ratings_df, features_df)
    elif args.model_name == 't5':
        print('Project is not supposed for T5 training')